import difflib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
BUNDLE_PATH = MODELS / "recommender.joblib"


def find_netflix_file():
    for f in sorted(RAW.rglob("*.csv")):
        if "netflix_titles" in f.name.lower():
            return f
    print("!! netflix_titles.csv not found in data/raw. CSV files I can see:")
    for f in sorted(RAW.rglob("*.csv")):
        print("   ", f.relative_to(RAW))
    raise SystemExit


def build():
    path = find_netflix_file()
    print("Using:", path.relative_to(ROOT))
    df = pd.read_csv(path)
    for c in ["title", "type", "listed_in", "description"]:
        if c not in df.columns:
            print(f"!! Missing column '{c}'. Columns found: {list(df.columns)}")
            raise SystemExit

    df = df.drop_duplicates(subset=["title", "type"]).reset_index(drop=True)
    df["listed_in"] = df["listed_in"].fillna("")
    df["description"] = df["description"].fillna("")

    # 'International Movies, Dramas' -> 'International_Movies Dramas'
    genre_tokens = df["listed_in"].apply(
        lambda s: " ".join(g.strip().replace(" ", "_").replace("&", "and")
                           for g in s.split(",") if g.strip()))
    df["text"] = genre_tokens + " " + genre_tokens + " " + df["description"]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=2, max_features=30000)
    X = vec.fit_transform(df["text"])
    print(f"Titles: {len(df):,}   Vocabulary size: {X.shape[1]:,}")

    meta = df[["title", "type", "release_year", "listed_in", "description"]].copy()
    meta["title_lower"] = meta["title"].str.lower().str.strip()
    bundle = {"vectorizer": vec, "matrix": X, "meta": meta}
    joblib.dump(bundle, BUNDLE_PATH)
    print("Saved:", BUNDLE_PATH)
    return bundle


_cache = {}
def load_bundle():
    if "b" not in _cache:
        _cache["b"] = joblib.load(BUNDLE_PATH)
    return _cache["b"]


def recommend(title, n=5, type_filter=None, bundle=None):
    """Return (matched_title, DataFrame of n similar titles).
    type_filter: 'Movie', 'TV Show' or None."""
    b = bundle or load_bundle()
    meta, X = b["meta"], b["matrix"]
    q = title.lower().strip()

    hits = meta.index[meta["title_lower"] == q].tolist()
    if not hits:
        close = difflib.get_close_matches(q, meta["title_lower"].tolist(), n=1, cutoff=0.6)
        if not close:
            return None, pd.DataFrame()
        hits = meta.index[meta["title_lower"] == close[0]].tolist()
    idx = hits[0]

    sims = linear_kernel(X[idx], X).ravel()
    sims[idx] = -1
    if type_filter:
        sims[(meta["type"] != type_filter).values] = -1
    top = np.argsort(sims)[::-1][:n]

    out = meta.iloc[top][["title", "type", "release_year", "listed_in"]].copy()
    out["similarity"] = sims[top].round(3)
    return meta.loc[idx, "title"], out.reset_index(drop=True)


def genre_overlap_score(bundle, sample=300, k=5, seed=42):
    """Simple evaluation: average genre overlap (Jaccard) between a title and its top-k results."""
    meta, X = bundle["meta"], bundle["matrix"]
    rng = np.random.default_rng(seed)
    gsets = meta["listed_in"].apply(lambda s: {g.strip() for g in s.split(",") if g.strip()})
    scores = []
    for i in rng.choice(len(meta), size=min(sample, len(meta)), replace=False):
        sims = linear_kernel(X[i], X).ravel()
        sims[i] = -1
        for j in np.argsort(sims)[::-1][:k]:
            a, b_ = gsets[i], gsets[j]
            if a | b_:
                scores.append(len(a & b_) / len(a | b_))
    return float(np.mean(scores))


if __name__ == "__main__":
    bundle = build()
    meta = bundle["meta"]

    tests = [t for t in ["Stranger Things", "The Crown", "Narcos", "Inception"]
             if t.lower() in set(meta["title_lower"])]
    tests += meta["title"].sample(2, random_state=1).tolist()

    for t in tests:
        matched, recs = recommend(t, n=5, bundle=bundle)
        print(f"\n=== Because you searched: {matched} ===")
        print(recs.to_string(index=False))

    score = genre_overlap_score(bundle)
    print(f"\nAverage genre overlap of top-5 recommendations: {score:.2f}")