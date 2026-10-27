import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean" / "platform_clean.csv")
sns.set_theme(style="whitegrid")

PLAT = {"Netflix": "Netflix", "Hulu": "Hulu",
        "Prime_Video": "Prime Video", "Disney_Plus": "Disney+"}

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("Saved", name)

# 1. Titles per platform
counts = {n: int(df[c].sum()) for c, n in PLAT.items()}
plt.figure(figsize=(7, 4))
sns.barplot(x=list(counts.keys()), y=list(counts.values()), color="#4C72B0")
plt.title("Number of titles per platform")
plt.ylabel("Titles")
save("01_titles_per_platform.png")

# 2. Movies vs TV split per platform
rows = []
for c, n in PLAT.items():
    sub = df[df[c] == 1]
    for t, k in sub["type"].value_counts().items():
        rows.append({"Platform": n, "Type": t, "Titles": k})
plt.figure(figsize=(7, 4))
sns.barplot(data=pd.DataFrame(rows), x="Platform", y="Titles", hue="Type")
plt.title("Movies vs TV shows by platform")
save("02_movies_vs_tv.png")

# 3. Top genres overall
g = df.dropna(subset=["genres"]).copy()
g["genre"] = g["genres"].str.split(",")
g = g.explode("genre")
g["genre"] = g["genre"].str.strip()
top = g["genre"].value_counts().head(12)
plt.figure(figsize=(8, 5))
sns.barplot(x=top.values, y=top.index, color="#55A868")
plt.title("Top 12 genres (all platforms)")
plt.xlabel("Titles")
save("03_top_genres.png")

# 4. Genre mix by platform (share of each platform's titles)
top8 = g["genre"].value_counts().head(8).index
mix = {}
for c, n in PLAT.items():
    sub = g[g[c] == 1]
    share = sub["genre"].value_counts(normalize=True) * 100
    mix[n] = share.reindex(top8).fillna(0)
plt.figure(figsize=(9, 5))
sns.heatmap(pd.DataFrame(mix), annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Genre mix by platform (% of that platform's genre tags)")
save("04_genre_mix_by_platform.png")

# 5. Titles by release decade
d = df.dropna(subset=["year"]).copy()
d = d[d["year"] >= 1950]
d["decade"] = (d["year"] // 10 * 10).astype(int)
plt.figure(figsize=(8, 4))
sns.countplot(data=d, x="decade", color="#C44E52")
plt.title("Titles by release decade")
save("05_titles_by_decade.png")

# 6. Rating distribution
r = df.dropna(subset=["imdb_rating"])
plt.figure(figsize=(7, 4))
sns.histplot(r["imdb_rating"], bins=30, color="#8172B2")
plt.axvline(7.0, color="red", linestyle="--", label="High-rating cut-off (7.0)")
plt.legend()
plt.title("IMDb rating distribution")
save("06_rating_distribution.png")

# 7. Average rating by platform
avg = {n: r.loc[r[c] == 1, "imdb_rating"].mean() for c, n in PLAT.items()}
plt.figure(figsize=(7, 4))
sns.barplot(x=list(avg.keys()), y=list(avg.values()), color="#CCB974")
plt.ylim(5, 7)
plt.title("Average IMDb rating by platform")
plt.ylabel("Average rating")
save("07_avg_rating_by_platform.png")

# 8. Share of high-rated titles by genre (genres with 200+ titles)
rg = g.dropna(subset=["high_rating"])
stats = rg.groupby("genre")["high_rating"].agg(["mean", "count"])
stats = stats[stats["count"] >= 200].sort_values("mean", ascending=False).head(12)
plt.figure(figsize=(8, 5))
sns.barplot(x=stats["mean"] * 100, y=stats.index, color="#64B5CD")
plt.title("Share of titles rated 7.0+ by genre")
plt.xlabel("% of titles rated 7.0 or higher")
save("08_high_rating_by_genre.png")

print("\nHigh-rating share overall:", round(df["high_rating"].mean() * 100, 1), "%")
print("Charts are in:", FIG)