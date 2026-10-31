import os
import sys
import tomllib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
import html
import plotly.io as pio
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from recommender import recommend, load_bundle  # noqa: E402
from mood import recommend_by_mood, MOODS  # noqa: E402

st.set_page_config(page_title="OTT Content Trends", page_icon="🎬", layout="wide")
pio.templates.default = "plotly_dark"
px.defaults.color_discrete_sequence = ["#E50914", "#4C9AFF", "#F5C518", "#2ECC71", "#B388FF", "#FF8A65"]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.5px;}
.hero {background: linear-gradient(120deg, #E50914 0%, #8E0E1A 45%, #1A1D26 100%); padding: 28px 32px; border-radius: 16px; margin-bottom: 22px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 6px 0 0; color: #f1f1f1; opacity: .9;}
.kpi {background: #1A1D26; border: 1px solid #2A2E3A; border-radius: 14px; padding: 18px 20px;}
.kpi .v {font-size: 1.9rem; font-weight: 700; color: #fff;}
.kpi .l {font-size: .8rem; color: #9AA0AE; text-transform: uppercase; letter-spacing: .06em;}
.card {background: #1A1D26; border: 1px solid #2A2E3A; border-radius: 12px; overflow: hidden; margin-bottom: 18px; transition: transform .15s, border-color .15s;}
.card:hover {transform: translateY(-4px); border-color: #E50914;}
.card img {width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block;}
.card .ph {width: 100%; aspect-ratio: 2/3; display: flex; align-items: center; justify-content: center; font-size: 3rem; background: #232736;}
.card .body {padding: 10px 12px 12px;}
.card .t {font-weight: 600; font-size: .92rem; line-height: 1.25; color: #fff; min-height: 2.4em;}
.card .m {font-size: .78rem; color: #9AA0AE; margin-top: 4px;}
.badge {display: inline-block; background: #F5C518; color: #111; font-weight: 700; font-size: .78rem; padding: 2px 8px; border-radius: 6px; margin-top: 6px;}
.pill {display: inline-block; background: #2A2E3A; color: #cfd3dc; font-size: .7rem; padding: 2px 8px; border-radius: 20px; margin: 5px 4px 0 0;}
.soon {text-align: center; padding: 60px 20px; border: 2px dashed #2A2E3A; border-radius: 16px; background: #14161D;}
.soon .big {font-size: 3.2rem;}
.soon h2 {margin: 10px 0 6px;}
.soon p {color: #9AA0AE; max-width: 560px; margin: 0 auto;}
[data-testid="stSidebar"] {background: #12141B;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>',
                 unsafe_allow_html=True)


def style(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, l=10, r=10, b=10))
    return fig


def coming_soon(icon, title, text, items):
    st.markdown(f'<div class="soon"><div class="big">{icon}</div><h2>{title}</h2>'
                f'<p>{text}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Planned for the next phase")
    for i in items:
        st.markdown(f"- {i}")

PLAT = {"Netflix": "Netflix", "Hulu": "Hulu", "Prime_Video": "Prime Video", "Disney_Plus": "Disney+"}
PLAT_REV = {v: k for k, v in PLAT.items()}

# TMDB genre names -> IMDb-style names used by the model
GENRE_MAP = {
    "Science Fiction": ["Sci-Fi"], "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"],
    "Action & Adventure": ["Action", "Adventure"], "War & Politics": ["War"],
    "Kids": ["Family"], "Soap": ["Drama"], "TV Movie": [], "Reality": ["Reality-TV"],
    "Talk": ["Talk-Show"], "News": ["News"],
}


# ---------------- data and models ----------------
@st.cache_data
def load_platforms():
    df = pd.read_csv(ROOT / "data" / "clean" / "platform_clean.csv")
    df["genre_list"] = df["genres"].fillna("").apply(
        lambda s: [g.strip() for g in s.split(",") if g.strip()])
    return df


@st.cache_data
def load_forecasts():
    fc = pd.read_csv(ROOT / "data" / "clean" / "forecasts.csv", parse_dates=["ds"])
    summ = pd.read_csv(ROOT / "data" / "clean" / "forecast_summary.csv")
    return fc, summ


@st.cache_resource
def load_models():
    m = ROOT / "models"
    models = {"Random Forest": joblib.load(m / "random_forest.joblib"),
              "XGBoost": joblib.load(m / "xgboost.joblib")}
    return models, joblib.load(m / "feature_columns.joblib")


def predict_probs(year, runtime, is_tv, plats, genres, age):
    models, cols = load_models()
    row = pd.DataFrame(0.0, index=[0], columns=cols)

    def put(c, v):
        if c in row.columns:
            row.loc[0, c] = v

    put("year", year)
    put("runtime", runtime)
    put("is_tv", int(is_tv))
    for p in plats:
        put(PLAT_REV[p], 1)
    for g in genres:
        put(f"genre_{g}", 1)
    put(f"age_{age}", 1)
    return {n: float(m.predict_proba(row)[0, 1]) for n, m in models.items()}


def show_prediction(probs):
    cols = st.columns(len(probs))
    for c, (name, p) in zip(cols, probs.items()):
        c.metric(f"{name}: chance of 7.0+", f"{p:.0%}")
    avg = sum(probs.values()) / len(probs)
    if avg >= 0.5:
        st.success("The models expect this title to be rated 7.0 or higher.")
    else:
        st.info("The models expect this title to be rated below 7.0.")
    st.caption("Based on genre, year, runtime, age rating and platform only. "
               "It is a statistical estimate, not a guarantee.")


# ---------------- TMDB ----------------
def tmdb_key():
    f = ROOT / ".streamlit" / "secrets.toml"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                k = tomllib.load(fh).get("TMDB_API_KEY")
            if k:
                return str(k).strip()
        except Exception:
            pass
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY")

@st.cache_data(ttl=3600)
def tmdb_get(path, **params):
    key = tmdb_key()
    if not key:
        return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/{path}",
                         params={"api_key": key, **params}, timeout=10)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


# ---------------- pages ----------------
def page_overview():
    hero(" OTT Content Trends and Prediction Dashboard",
         "Trends, forecasts and rating predictions across Netflix, Prime Video, Hulu and Disney+")
    df = load_platforms()
    c = st.columns(4)
    kpi(c[0], "Total titles", f"{len(df):,}")
    kpi(c[1], "Movies", f"{(df['type'] == 'Movie').sum():,}")
    kpi(c[2], "TV shows", f"{(df['type'] == 'TV Show').sum():,}")
    kpi(c[3], "Platforms", len(PLAT))
    st.write("")

    left, right = st.columns(2)
    g = df.explode("genre_list")["genre_list"].value_counts().head(12).reset_index()
    g.columns = ["Genre", "Titles"]
    left.plotly_chart(style(px.bar(g.sort_values("Titles"), x="Titles", y="Genre",
                                   orientation="h", title="Top 12 genres")))
    split = df["type"].value_counts().reset_index()
    split.columns = ["Type", "Titles"]
    right.plotly_chart(style(px.pie(split, names="Type", values="Titles",
                                    title="Movies vs TV shows", hole=0.5)))

    per = pd.DataFrame({"Platform": list(PLAT.values()),
                        "Titles": [int(df[c_].sum()) for c_ in PLAT]})
    st.plotly_chart(style(px.bar(per, x="Platform", y="Titles", color="Platform",
                                 title="Titles per platform")))
    st.caption("Platform availability is a 2020-21 snapshot. A title on several platforms counts once for each.")


def page_trends():
    st.title(" Trends and Platform Viewer")
    tab1, tab2 = st.tabs(["Forecasts", "Platform comparison"])

    with tab1:
        fc, summ = load_forecasts()
        series = st.selectbox("Genre or series", sorted(fc["series"].unique()))
        d = fc[fc["series"] == series].sort_values("ds").reset_index(drop=True)
        n = 3 if series.startswith("Platform") else 36
        d["is_fc"] = False
        d.loc[d.index[-n:], "is_fc"] = True
        lo, hi = int(d["ds"].dt.year.min()), int(d["ds"].dt.year.max())
        yr = st.slider("Year range", lo, hi, (lo, hi))
        m = d[(d["ds"].dt.year >= yr[0]) & (d["ds"].dt.year <= yr[1])]
        hist, fut = m[~m["is_fc"]], m[m["is_fc"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], name="Model fit",
                                 line=dict(color="#4C72B0")))
        if not fut.empty:
            fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat_upper"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat_lower"], fill="tonexty",
                                     fillcolor="rgba(196,78,82,0.2)", line=dict(width=0),
                                     name="Uncertainty range"))
            fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast",
                                     line=dict(color="#C44E52", dash="dash")))
        fig.update_layout(title=f"{series}: titles added per period", yaxis_title="Titles")
        st.plotly_chart(fig)

        row = summ[summ["Series"].str.contains(series.replace("Platform: ", ""), regex=False)]
        if not row.empty:
            r = row.iloc[0]
            st.write(f"**Trend:** {r['Trend']} ({r['Change_%']}% vs the last 12 months)")
        with st.expander("Forecast summary for all series"):
            st.dataframe(summ)
        st.caption("Netflix series are monthly counts of titles added. Platform series use yearly "
                   "release counts, so treat them as indicative only.")

    with tab2:
        df = load_platforms()
        rated = df.dropna(subset=["imdb_rating"])
        avg = pd.DataFrame({"Platform": list(PLAT.values()),
                            "Average IMDb rating": [rated.loc[rated[c] == 1, "imdb_rating"].mean()
                                                    for c in PLAT]})
        st.plotly_chart(px.bar(avg, x="Platform", y="Average IMDb rating",
                               title="Average IMDb rating by platform", range_y=[5, 7.5]))
        g = df.explode("genre_list")
        top8 = g["genre_list"].value_counts().head(8).index
        mix = pd.DataFrame({name: (g[g[c] == 1]["genre_list"].value_counts(normalize=True) * 100)
                            .reindex(top8).fillna(0) for c, name in PLAT.items()})
        st.plotly_chart(px.imshow(mix.round(1), text_auto=True, aspect="auto",
                                  color_continuous_scale="YlGnBu",
                                  title="Genre mix by platform (% of genre tags)"))


def page_rating():
    st.title(" Title Rating Lookup and Prediction")
    models, cols = load_models()
    genre_opts = sorted(c[6:] for c in cols if c.startswith("genre_"))
    age_opts = sorted(c[4:] for c in cols if c.startswith("age_"))

    tab1, tab2 = st.tabs(["Search a real title (TMDB)", "Predict a new title"])

    with tab1:
        if not tmdb_key():
            st.warning("No TMDB API key found. Add it to .streamlit/secrets.toml to use live search.")
        q = st.text_input("Search any movie or TV show")
        if q and tmdb_key():
            data = tmdb_get("search/multi", query=q)
            results = [r for r in (data or {}).get("results", [])
                       if r.get("media_type") in ("movie", "tv")][:8]
            if not results:
                st.info("No results. Try another title.")
            else:
                def label(r):
                    t = r.get("title") or r.get("name")
                    y = (r.get("release_date") or r.get("first_air_date") or "")[:4]
                    return f"{t} ({y or '?'}), {r['media_type']}"
                pick = st.selectbox("Pick a result", results, format_func=label)
                det = tmdb_get(f"{pick['media_type']}/{pick['id']}") or pick
                is_tv = pick["media_type"] == "tv"
                title = det.get("title") or det.get("name")
                c1, c2 = st.columns([1, 3])
                if det.get("poster_path"):
                    c1.image("https://image.tmdb.org/t/p/w300" + det["poster_path"], width=220)
                c2.subheader(title)
                c2.write(det.get("overview", "No overview available."))
                a, b = c2.columns(2)
                a.metric("TMDB audience rating", f"{det.get('vote_average', 0):.1f} / 10")
                b.metric("Votes", f"{det.get('vote_count', 0):,}")
                c2.caption("TMDB ratings come from TMDB users and can differ from IMDb scores.")

                names = [g["name"] for g in det.get("genres", [])]
                mapped = []
                for nme in names:
                    mapped += GENRE_MAP.get(nme, [nme])
                mapped = [g for g in dict.fromkeys(mapped) if g in genre_opts]
                year = int((det.get("release_date") or det.get("first_air_date") or "2020")[:4])
                runtime = det.get("runtime") or ((det.get("episode_run_time") or [45])[0])

                st.markdown("### Model prediction")
                st.write(f"Using: {', '.join(mapped) or 'no matching genres'} | {year} | {runtime} min")
                show_prediction(predict_probs(year, runtime, is_tv, ["Netflix"], mapped, "Unknown"))

                matched, recs = recommend(title, n=5)
                if matched:
                    st.markdown(f"### Similar titles (closest catalogue match: {matched})")
                    st.dataframe(recs[["title", "type", "release_year", "listed_in", "similarity"]])
                else:
                    st.caption("This title isn't in the Netflix catalogue, so no similar titles are shown.")

    with tab2:
        st.write("Describe a title that doesn't exist yet and see what the models expect.")
        c1, c2 = st.columns(2)
        kind = c1.radio("Type", ["Movie", "TV Show"], horizontal=True)
        year = c1.slider("Release year", 1980, 2030, 2025)
        runtime = c1.slider("Runtime (minutes, per episode for TV)", 15, 240, 100 if kind == "Movie" else 45)
        genres = c2.multiselect("Genres", genre_opts, default=["Drama"] if "Drama" in genre_opts else [])
        age = c2.selectbox("Age rating", age_opts, index=age_opts.index("Unknown") if "Unknown" in age_opts else 0)
        plats = c2.multiselect("Platforms", list(PLAT.values()), default=["Netflix"])
        if st.button("Predict"):
            show_prediction(predict_probs(year, runtime, kind == "TV Show", plats, genres, age))

    with st.expander("Model comparison on the test set"):
        comp = pd.read_csv(ROOT / "models" / "model_comparison.csv")
        st.dataframe(comp)
        st.caption("Baseline accuracy (always guessing 'not high') is about 61.9%.")


def page_mood():
    st.title(" Mood-Based Discovery")
    labels = {f"{d['emojis'][0]} {m.title()}": m for m, d in MOODS.items()}
    choice = st.radio("How are you feeling?", list(labels), horizontal=True)
    text = st.text_input("...or type a mood or a few words (optional, overrides the choice above)")
    kind = st.selectbox("Show me", ["Anything", "Movie", "TV Show"])

    if st.button("Suggest something"):
        user_input = text.strip() or labels[choice]
        mood, recs = recommend_by_mood(user_input, n=5,
                                       type_filter=None if kind == "Anything" else kind)
        if recs.empty:
            st.info("I couldn't find a match for that. Try another mood or different words.")
        else:
            st.write(f"Mood detected: **{mood or 'based on your words'}**")
            for _, r in recs.iterrows():
                st.markdown(f"**{r['title']}** ({r['type']}, {int(r['release_year'])})  \n"
                            f"*{r['listed_in']}*  \n{r['description']}")
                st.divider()
    st.caption("This uses a hand-made mood-to-genre mapping plus TF-IDF ranking. "
               "Nothing about you is stored.")


IMG_BASE = "https://image.tmdb.org/t/p/w300"


def _fetch_poster(key, imdb_id, title, year, is_tv):
    """Plain function (no Streamlit calls) so it can run in threads."""
    try:
        if imdb_id.startswith("tt"):
            r = requests.get(f"https://api.themoviedb.org/3/find/{imdb_id}",
                             params={"api_key": key, "external_source": "imdb_id"},
                             timeout=8).json()
            for k in ("movie_results", "tv_results"):
                for item in r.get(k, []):
                    if item.get("poster_path"):
                        return IMG_BASE + item["poster_path"]
        # fallback: search by title (and year for movies)
        params = {"api_key": key, "query": title}
        if year and not is_tv:
            params["year"] = int(year)
        ep = "search/tv" if is_tv else "search/movie"
        r = requests.get(f"https://api.themoviedb.org/3/{ep}", params=params, timeout=8).json()
        for item in r.get("results", [])[:3]:
            if item.get("poster_path"):
                return IMG_BASE + item["poster_path"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_posters(items, key):
    """items: tuple of (imdb_id, title, year, is_tv). Returns a list of poster URLs or None."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(lambda a: _fetch_poster(key, *a), items))


def card_html(r, url):
    t = html.escape(str(r.title))
    img = f'<img src="{url}" alt="{t}">' if url else '<div class="ph"></div>'
    year = int(r.year) if pd.notna(r.year) else "?"
    rating = f'<span class="badge"> {r.imdb_rating:.1f}</span>' if pd.notna(r.imdb_rating) else ""
    genres = r.genres if isinstance(r.genres, str) else ""
    plats = r.platforms if isinstance(r.platforms, str) else ""
    gp = "".join(f'<span class="pill">{html.escape(x.strip())}</span>'
                 for x in genres.split(",")[:2] if x.strip())
    pp = "".join(f'<span class="pill">{html.escape(x.strip())}</span>'
                 for x in plats.split(",") if x.strip())
    return (f'<div class="card">{img}<div class="body"><div class="t">{t}</div>'
            f'<div class="m">{year} · {r.type}</div>{rating}<div>{gp}</div><div>{pp}</div></div></div>')




WALL_CSS = """<style>
.wall{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;margin-top:10px}
.wall .p{background:#1A1D26;border:1px solid #2A2E3A;border-radius:12px;overflow:hidden;transition:transform .15s,border-color .15s}
.wall .p:hover{transform:translateY(-4px);border-color:#E50914}
.wall img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block}
.wall .c{padding:8px 10px}
.wall .n{font-weight:600;font-size:.85rem;color:#fff;line-height:1.2}
.wall .y{font-size:.75rem;color:#9AA0AE;margin-top:2px}
</style>"""


@st.cache_data(ttl=86400, show_spinner=False)
def get_details(imdb_id, title, year, is_tv, key):
    base = "https://api.themoviedb.org/3"
    try:
        tmdb_id, media = None, None
        if imdb_id.startswith("tt"):
            r = requests.get(f"{base}/find/{imdb_id}",
                             params={"api_key": key, "external_source": "imdb_id"},
                             timeout=8).json()
            order = ((("tv_results", "tv"), ("movie_results", "movie")) if is_tv
                     else (("movie_results", "movie"), ("tv_results", "tv")))
            for k, m in order:
                if r.get(k):
                    tmdb_id, media = r[k][0]["id"], m
                    break
        if tmdb_id is None:
            media = "tv" if is_tv else "movie"
            params = {"api_key": key, "query": title}
            if year and not is_tv:
                params["year"] = int(year)
            r = requests.get(f"{base}/search/{media}", params=params, timeout=8).json()
            if r.get("results"):
                tmdb_id = r["results"][0]["id"]
        if tmdb_id is None:
            return None

        d = requests.get(f"{base}/{media}/{tmdb_id}",
                         params={"api_key": key, "append_to_response": "credits"},
                         timeout=8).json()
        credits = d.get("credits", {})
        cast = [c["name"] for c in credits.get("cast", [])[:6]]
        if media == "movie":
            makers = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"][:2]
            runtime = d.get("runtime")
            maker_label = "Director"
        else:
            makers = [c["name"] for c in d.get("created_by", [])][:2]
            ert = d.get("episode_run_time") or []
            runtime = ert[0] if ert else None
            maker_label = "Created by"
        return {"overview": d.get("overview") or "", "tagline": d.get("tagline") or "",
                "tmdb_rating": d.get("vote_average"), "votes": d.get("vote_count"),
                "runtime": runtime, "genres": [g["name"] for g in d.get("genres", [])],
                "cast": cast, "makers": makers, "maker_label": maker_label}
    except Exception:
        return None


@st.dialog("Title details", width="large")
def show_details(info, key):
    d = get_details(info["imdb_id"], info["title"], info["year"], info["is_tv"], key)
    left, right = st.columns([1, 2])
    left.image(info["poster"])
    with right:
        st.subheader(f"{info['title']} ({info['year'] or '?'})")
        st.caption(f"{info['type']} · {info['platforms'] or 'Platform unknown'}")
        if d and d["tagline"]:
            st.markdown(f"*{d['tagline']}*")

        a, b, c = st.columns(3)
        a.metric("IMDb rating", f"{info['imdb_rating']:.1f}" if info["imdb_rating"] is not None else "n/a")
        if d and d["tmdb_rating"]:
            b.metric("TMDB rating", f"{d['tmdb_rating']:.1f}")
        if d and d["runtime"]:
            c.metric("Runtime", f"{d['runtime']} min")

        st.markdown("#### Story")
        if d and d["overview"]:
            st.write(d["overview"])
        else:
            st.write("No story is available for this title on TMDB.")

        genres = ", ".join(d["genres"]) if d and d["genres"] else (info["genres"] or "")
        if genres:
            st.write(f"**Genres:** {genres}")
        if d and d["makers"]:
            st.write(f"**{d['maker_label']}:** {', '.join(d['makers'])}")
        if d and d["cast"]:
            st.write(f"**Cast:** {', '.join(d['cast'])}")

    matched, recs = recommend(info["title"], n=5)
    if matched is not None and not recs.empty:
        st.markdown("#### You might also like")
        st.write(" · ".join(f"{t}" for t in recs["title"]))


def page_browse():
    hero(" Browse and Explore", "Filter the catalogue, then click Story under any poster")
    df = load_platforms()
    c1, c2, c3, c4 = st.columns(4)
    q = c1.text_input("Title contains")
    kind = c2.selectbox("Type", ["All", "Movie", "TV Show"])
    all_genres = sorted({g for gl in df["genre_list"] for g in gl})
    genre = c3.selectbox("Genre", ["All"] + all_genres)
    plat = c4.selectbox("Platform", ["All"] + list(PLAT.values()))
    lo, hi = st.slider("IMDb rating", 0.0, 10.0, (0.0, 10.0), step=0.1)

    f = df.copy()
    if q:
        f = f[f["title"].str.contains(q, case=False, na=False)]
    if kind != "All":
        f = f[f["type"] == kind]
    if genre != "All":
        f = f[f["genre_list"].apply(lambda gl: genre in gl)]
    if plat != "All":
        f = f[f[PLAT_REV[plat]] == 1]
    if (lo, hi) != (0.0, 10.0):
        f = f[f["imdb_rating"].between(lo, hi)]
    f = f.sort_values("imdb_rating", ascending=False).reset_index(drop=True)

    st.write(f"**{len(f):,}** titles match")
    if f.empty:
        st.info("No titles match these filters.")
        return

    key = tmdb_key()
    if not key:
        secrets_file = ROOT / ".streamlit" / "secrets.toml"
        st.error("TMDB key not found. The app looked for it here:")
        st.code(str(secrets_file))
        return

    per_page = 40
    pages = max(1, -(-len(f) // per_page))
    page = st.number_input("Page", 1, pages, 1)
    chunk = f.iloc[(page - 1) * per_page: page * per_page]
    st.caption(f"Page {page} of {pages}")

    items = tuple((str(r.imdb_id) if pd.notna(r.imdb_id) else "",
                   str(r.title),
                   int(r.year) if pd.notna(r.year) else 0,
                   r.type == "TV Show") for r in chunk.itertuples())
    with st.spinner("Loading posters..."):
        urls = get_posters(items, key)

    shown = [(r, u) for r, u in zip(chunk.itertuples(), urls) if u]
    if not shown:
        st.info("No posters came back for this page. Try another page or filter.")
        return

    for start in range(0, len(shown), 5):
        cols = st.columns(5)
        for col, (r, u) in zip(cols, shown[start:start + 5]):
            with col:
                year = int(r.year) if pd.notna(r.year) else "?"
                rating = f" ·  {r.imdb_rating:.1f}" if pd.notna(r.imdb_rating) else ""
                st.image(u)
                st.markdown(f"**{r.title}**")
                st.caption(f"{year}{rating}")
                if st.button(" Story", key=f"story_{page}_{r.Index}"):
                    show_details({
                        "title": str(r.title),
                        "year": int(r.year) if pd.notna(r.year) else 0,
                        "is_tv": r.type == "TV Show",
                        "type": r.type,
                        "imdb_id": str(r.imdb_id) if pd.notna(r.imdb_id) else "",
                        "imdb_rating": float(r.imdb_rating) if pd.notna(r.imdb_rating) else None,
                        "genres": r.genres if isinstance(r.genres, str) else "",
                        "platforms": r.platforms if isinstance(r.platforms, str) else "",
                        "poster": u,
                    }, key)
# ---------------- navigation ----------------
def page_rating_soon():
    hero(" Title Rating Lookup", "Search any real title and compare it with the model's prediction")
    coming_soon("🚧", "Coming in the next phase",
                "Live TMDB search and the interactive rating predictor will be added once "
                "the core analysis is complete.",
                ["Live TMDB search with poster, overview and audience rating",
                 "Random Forest and XGBoost prediction shown beside the real rating",
                 "Similar-title recommendations for any searched title",
                 "A form to describe a brand-new title and predict its rating"])


def page_mood_soon():
    hero(" Mood-Based Discovery", "Tell us how you feel and get a suggestion")
    coming_soon("🚧", "Coming in the next phase",
                "The recommendation engine is built and tested. Its dashboard page is being "
                "polished for the next release.",
                ["Pick an emoji or type a mood to get a matching title",
                 "Poster cards for every suggestion",
                 "Movie or TV filter and variety on every click"])

NETFLIX_TO_MODEL = [
    ("drama", ["Drama"]), ("comed", ["Comedy"]), ("stand-up", ["Comedy"]),
    ("thriller", ["Thriller"]), ("horror", ["Horror"]),
    ("action", ["Action", "Adventure"]), ("adventure", ["Adventure"]),
    ("sci-fi", ["Sci-Fi", "Fantasy"]), ("fantasy", ["Fantasy"]),
    ("romantic", ["Romance"]), ("documentar", ["Documentary"]),
    ("docuseries", ["Documentary"]), ("crime", ["Crime"]),
    ("mystery", ["Mystery"]), ("mysteries", ["Mystery"]),
    ("family", ["Family"]), ("children", ["Family"]), ("kids", ["Family"]),
    ("anime", ["Animation"]), ("music", ["Music"]), ("reality", ["Reality-TV"]),
    ("sport", ["Sport"]),
]


def infer_genres(story, bundle, top_k=15):
    """Find the catalogue titles closest to the story and vote on their genres."""
    vec, X, meta = bundle["vectorizer"], bundle["matrix"], bundle["meta"]
    sims = linear_kernel(vec.transform([story]), X).ravel()
    idx = np.argsort(sims)[::-1][:top_k]
    weights = {}
    for i in idx:
        if sims[i] <= 0:
            continue
        found = set()
        for g in str(meta.iloc[i]["listed_in"]).lower().split(","):
            for key, mapped in NETFLIX_TO_MODEL:
                if key in g:
                    found.update(mapped)
        for m in found:
            weights[m] = weights.get(m, 0) + float(sims[i])
    return weights, idx, sims


def page_predictor():
    hero(" Pitch Predictor", "Describe a new story in a few lines and see how it might be rated")
    models, cols = load_models()
    genre_opts = sorted(c[6:] for c in cols if c.startswith("genre_"))
    age_opts = sorted(c[4:] for c in cols if c.startswith("age_"))

    story = st.text_area(
        "Your story (2 to 3 lines)", height=110,
        placeholder="A retired detective in Mumbai is pulled back into a case when his "
                    "former partner disappears, forcing him to face a conspiracy inside the police.")
    c1, c2 = st.columns(2)
    kind = c1.radio("Type", ["Movie", "TV Show"], horizontal=True)
    year = c1.slider("Release year", 1980, 2030, 2026)
    runtime = c1.slider("Runtime (minutes, per episode for TV)", 15, 240,
                        110 if kind == "Movie" else 45)
    plats = c2.multiselect("Platforms", list(PLAT.values()), default=["Netflix"])
    age = c2.selectbox("Age rating", age_opts,
                       index=age_opts.index("Unknown") if "Unknown" in age_opts else 0)
    manual = c2.multiselect("Override genres (optional, leave empty to detect from the story)",
                            genre_opts)

    if not st.button(" Predict outcome", type="primary"):
        return
    if len(story.split()) < 8 and not manual:
        st.warning("Please write at least a couple of full sentences so the story can be matched.")
        return

    bundle = load_bundle()
    weights, idx, sims = infer_genres(story, bundle)
    if manual:
        genres = manual
    else:
        ranked = [g for g, _ in sorted(weights.items(), key=lambda kv: -kv[1]) if g in genre_opts]
        genres = ranked[:3]
    if not genres:
        st.warning("I couldn't match this story to any genre. Add more detail, "
                   "or choose genres manually with the override box.")
        return

    st.subheader("Prediction")
    st.write(f"**Genres used:** {', '.join(genres)}")
    show_prediction(predict_probs(year, runtime, kind == "TV Show", plats or ["Netflix"], genres, age))

    meta = bundle["meta"]
    near = meta.iloc[idx[:10]][["title", "type", "release_year", "listed_in"]].copy()
    near["similarity"] = sims[idx[:10]].round(3)
    near = near[near["similarity"] > 0]
    if not near.empty:
        rated = load_platforms().dropna(subset=["imdb_rating"]).copy()
        rated["k"] = rated["title"].str.lower().str.strip()
        lookup = rated.drop_duplicates("k").set_index("k")["imdb_rating"]
        near["imdb_rating"] = meta.loc[near.index, "title_lower"].map(lookup)

        st.subheader("Most similar existing titles")
        got = near["imdb_rating"].dropna()
        if len(got):
            a, b = st.columns(2)
            a.metric("Average IMDb rating of similar titles", f"{got.mean():.1f}")
            b.metric("Share rated 7.0 or higher", f"{(got >= 7).mean():.0%}")
        st.dataframe(near.rename(columns={"listed_in": "genres"}).reset_index(drop=True))

    st.caption("This is a statistical estimate from the genre your story falls into plus the "
               "details you chose. It does not judge writing quality, casting or marketing.")
# ---------------- navigation ----------------
PAGES = {":material/dashboard: Overview": page_overview,
         ":material/trending_up: Trends": page_trends,
         ":material/movie: Browse": page_browse,
         ":material/auto_awesome: Pitch Predictor": page_predictor,
         ":material/mood: Mood (soon)": page_mood_soon}

st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}

.st-key-topbar {position: sticky; top: 0; z-index: 999; padding: 10px 4px;
    margin-bottom: 18px; background: rgba(14,17,23,.92); backdrop-filter: blur(10px);
    border-bottom: 1px solid #2A2E3A;}
.brand {font-size: 1.3rem; font-weight: 800; color: #fff; letter-spacing: -.5px; padding-top: 4px;}
.brand span {color: #E50914;}

.st-key-nav div[role="radiogroup"] {gap: 6px; flex-wrap: wrap; justify-content: flex-end;}
.st-key-nav label[data-baseweb="radio"] {background: transparent; border: 1px solid transparent;
    padding: 7px 16px; border-radius: 999px; margin: 0; cursor: pointer;}
.st-key-nav label[data-baseweb="radio"] > div:first-child {display: none;}
.st-key-nav label[data-baseweb="radio"]:hover {background: #1A1D26;}
.st-key-nav label[data-baseweb="radio"]:has(input:checked) {background: #E50914;}
.st-key-nav label[data-baseweb="radio"]:has(input:checked) p {color: #fff; font-weight: 600;}

@media (max-width: 640px) {
  .st-key-nav div[role="radiogroup"] {justify-content: flex-start;}
  .st-key-nav label[data-baseweb="radio"] {padding: 6px 12px;}
}
</style>
""", unsafe_allow_html=True)

with st.container(key="topbar"):
    left, right = st.columns([1, 3])
    left.markdown('<div class="brand"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#E50914" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-6px;margin-right:6px"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M10 9l5 3-5 3z" fill="#E50914"/></svg><span>OTT</span> Insights</div>', unsafe_allow_html=True)
    with right:
        choice = st.radio("Navigate", list(PAGES), horizontal=True,
                          label_visibility="collapsed", key="nav")

PAGES[choice]()
