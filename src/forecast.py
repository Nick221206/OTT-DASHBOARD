import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CLEAN.mkdir(exist_ok=True)

HORIZON = 36   # months to forecast
START = "2016-01-01"


def find_netflix_file():
    for f in sorted(RAW.rglob("*.csv")):
        if "netflix_titles" in f.name.lower():
            return f
    print("!! netflix_titles.csv not found in data/raw")
    raise SystemExit


# ---------- load Netflix ----------
nf = pd.read_csv(find_netflix_file())
nf["date_added"] = pd.to_datetime(nf["date_added"].astype(str).str.strip(), errors="coerce")
nf = nf.dropna(subset=["date_added"])
last_month = nf["date_added"].max().to_period("M").to_timestamp()
end = last_month - pd.DateOffset(months=1)      # last month is probably incomplete
print(f"Titles with a date_added: {len(nf):,}")
print(f"Data runs to {last_month:%b %Y}; using months up to {end:%b %Y}")


def monthly(frame):
    s = frame.set_index("date_added").resample("MS").size()
    s = s.reindex(pd.date_range(START, end, freq="MS"), fill_value=0)
    return s


def make_model():
    return Prophet(yearly_seasonality=True, weekly_seasonality=False,
                   daily_seasonality=False, interval_width=0.8,
                   changepoint_prior_scale=0.1)


def forecast(series, periods, freq="MS"):
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    m = make_model() if freq == "MS" else Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        interval_width=0.8)
    m.fit(df)
    fc = m.predict(m.make_future_dataframe(periods=periods, freq=freq))
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[c] = fc[c].clip(lower=0)
    return fc


def backtest(series, holdout=12):
    """Hide the last `holdout` months, forecast them, compare with a naive guess."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = make_model()
    m.fit(df)
    pred = m.predict(m.make_future_dataframe(periods=holdout, freq="MS"))["yhat"].iloc[-holdout:].clip(lower=0)
    mae = float(np.mean(np.abs(test.values - pred.values)))
    naive = float(np.mean(np.abs(test.values - train.iloc[-12:].mean())))
    return mae, naive


def trend_label(series, fc):
    recent = series.iloc[-12:].mean()
    future = fc["yhat"].iloc[-HORIZON:-HORIZON + 12].mean()
    change = (future - recent) / recent * 100 if recent > 0 else np.nan
    if np.isnan(change):
        return "n/a", change
    return ("Growing" if change > 10 else "Declining" if change < -10 else "Stable"), change


def plot_series(ax, series, fc, title):
    ax.plot(series.index, series.values, color="#4C72B0", lw=1.3, label="Actual")
    f = fc[fc["ds"] > series.index[-1]]
    ax.plot(f["ds"], f["yhat"], color="#C44E52", lw=1.5, label="Forecast")
    ax.fill_between(f["ds"], f["yhat_lower"], f["yhat_upper"], color="#C44E52", alpha=0.2)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


rows, all_fc = [], []

# ---------- 1. overall ----------
overall = monthly(nf)
fc = forecast(overall, HORIZON)
mae, naive = backtest(overall)
label, change = trend_label(overall, fc)
rows.append({"Series": "All titles", "Trend": label, "Change_%": round(change, 1),
             "MAE": round(mae, 2), "Naive_MAE": round(naive, 2)})
all_fc.append(fc.assign(series="All titles")[["series", "ds", "yhat", "yhat_lower", "yhat_upper"]])

plt.figure(figsize=(10, 4))
plot_series(plt.gca(), overall, fc, "Netflix titles added per month: forecast")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "10_forecast_overall.png", dpi=150)
plt.close()

# ---------- 2. top genres ----------
g = nf.copy()
g["genre"] = g["listed_in"].fillna("").str.split(",")
g = g.explode("genre")
g["genre"] = g["genre"].str.strip()
g = g[g["genre"] != ""]
top_genres = g["genre"].value_counts().head(8).index.tolist()

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
for ax, genre in zip(axes.ravel(), top_genres):
    s = monthly(g[g["genre"] == genre])
    fc = forecast(s, HORIZON)
    mae, naive = backtest(s)
    label, change = trend_label(s, fc)
    rows.append({"Series": genre, "Trend": label, "Change_%": round(change, 1),
                 "MAE": round(mae, 2), "Naive_MAE": round(naive, 2)})
    all_fc.append(fc.assign(series=genre)[["series", "ds", "yhat", "yhat_lower", "yhat_upper"]])
    plot_series(ax, s, fc, f"{genre}  ({label})")
plt.tight_layout()
plt.savefig(FIG / "11_forecast_genres.png", dpi=150)
plt.close()

# ---------- 3. platforms (yearly release counts, experimental) ----------
pl = pd.read_csv(CLEAN / "platform_clean.csv")
pl = pl.dropna(subset=["year"])
max_year = int(pl["year"].max())
years = list(range(2000, max_year))      # drop the last year: likely incomplete
plat_cols = {"Netflix": "Netflix", "Hulu": "Hulu", "Prime_Video": "Prime Video", "Disney_Plus": "Disney+"}

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (c, name) in zip(axes.ravel(), plat_cols.items()):
    counts = pl[pl[c] == 1]["year"].astype(int).value_counts()
    s = pd.Series([counts.get(y, 0) for y in years],
                  index=pd.to_datetime([f"{y}-01-01" for y in years]))
    fc = forecast(s, 3, freq="YS")
    recent = s.iloc[-3:].mean()
    future = fc["yhat"].iloc[-3:].mean()
    change = (future - recent) / recent * 100 if recent > 0 else np.nan
    label = "Growing" if change > 10 else "Declining" if change < -10 else "Stable"
    rows.append({"Series": f"Platform: {name} (yearly)", "Trend": label,
                 "Change_%": round(change, 1), "MAE": np.nan, "Naive_MAE": np.nan})
    all_fc.append(fc.assign(series=f"Platform: {name}")[["series", "ds", "yhat", "yhat_lower", "yhat_upper"]])
    ax.plot(s.index, s.values, color="#4C72B0", marker="o", ms=3, label="Actual")
    f = fc[fc["ds"] > s.index[-1]]
    ax.plot(f["ds"], f["yhat"], color="#C44E52", marker="o", ms=3, label="Forecast")
    ax.fill_between(f["ds"], f["yhat_lower"], f["yhat_upper"], color="#C44E52", alpha=0.2)
    ax.set_title(f"{name}: titles by release year ({label})", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "12_forecast_platforms.png", dpi=150)
plt.close()

# ---------- save + summary ----------
summary = pd.DataFrame(rows)
summary.to_csv(CLEAN / "forecast_summary.csv", index=False)
pd.concat(all_fc).to_csv(CLEAN / "forecasts.csv", index=False)

print("\n", summary.to_string(index=False))
print("\nSaved: figures/10_forecast_overall.png, 11_forecast_genres.png, 12_forecast_platforms.png")
print("Saved: data/clean/forecasts.csv and forecast_summary.csv")