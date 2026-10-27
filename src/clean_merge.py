import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def make_key(s):
    """'The Office (US)!' -> 'theofficeus' : used to match titles across files."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())

def need(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"\n!! {name} is missing columns: {missing}")
        print("   Columns it actually has:", list(df.columns))
        raise SystemExit

def to_number(series):
    """'8.8/10' -> 8.8 , '98/100' -> 98 , 7.5 -> 7.5"""
    return pd.to_numeric(series.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")

# ---------- IMDb ----------
print("Loading IMDb files (1-3 minutes)...")
ratings = pd.read_csv(RAW / "title.ratings.tsv.gz", sep="\t")
keep = {"movie", "tvMovie", "tvSeries", "tvMiniSeries"}
parts = []
for ch in pd.read_csv(
    RAW / "title.basics.tsv.gz", sep="\t", na_values="\\N",
    usecols=["tconst", "titleType", "primaryTitle", "originalTitle",
             "startYear", "runtimeMinutes", "genres"],
    chunksize=500_000, low_memory=False,
):
    parts.append(ch[ch["titleType"].isin(keep)])
basics = pd.concat(parts).merge(ratings, on="tconst", how="inner")
basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce")
basics["imdb_is_tv"] = basics["titleType"].isin(["tvSeries", "tvMiniSeries"])

a = basics.assign(key=basics["primaryTitle"].map(make_key))
b = basics.assign(key=basics["originalTitle"].map(make_key))
imdb = pd.concat([a, b]).drop_duplicates(["tconst", "key"])
imdb = imdb[imdb["key"] != ""]
imdb = imdb[["tconst", "key", "startYear", "imdb_is_tv", "averageRating",
             "numVotes", "runtimeMinutes", "genres"]]
print(f"  IMDb titles with ratings: {len(imdb):,}")

def attach_imdb(df, year_col, is_tv_col):
    """Match each row to an IMDb title by cleaned title + close year + same kind."""
    df = df.copy()
    df["_row"] = np.arange(len(df))
    m = df[["_row", "title_key", year_col, is_tv_col]].merge(
        imdb, left_on="title_key", right_on="key", how="inner")
    m = m[m[is_tv_col] == m["imdb_is_tv"]]
    m["diff"] = (m[year_col] - m["startYear"]).abs()
    limit = np.where(m[is_tv_col], 15, 1)
    m = m[m["diff"] <= limit]
    m = m.sort_values(["_row", "diff", "numVotes"], ascending=[True, True, False])
    m = m.drop_duplicates("_row")
    m = m[["_row", "tconst", "averageRating", "numVotes", "runtimeMinutes", "genres"]]
    m.columns = ["_row", "imdb_id", "imdb_rating_file", "imdb_votes",
                 "imdb_runtime", "imdb_genres"]
    return df.merge(m, on="_row", how="left").drop(columns="_row")

# ---------- PLATFORMS (movies + TV) ----------
print("\nCleaning platform datasets...")
def find_file(*keywords):
    """Find a file in data/raw whose name contains all the keywords (case-insensitive)."""
    for f in sorted(RAW.rglob("*.csv")):
        n = f.name.lower()
        if all(k in n for k in keywords):
            print("  Using:", f.relative_to(RAW))
            return f
    print(f"\n!! No CSV found with {keywords} in its name.")
    print("   CSV files I can see in data/raw:")
    for f in sorted(RAW.rglob("*.csv")):
        print("    ", f.relative_to(RAW))
    raise SystemExit

mv = pd.read_csv(find_file("movie"))
tv = pd.read_csv(find_file("tv"))
must_have = ["Title", "Year", "Netflix", "Hulu", "Prime Video", "Disney+"]
need(mv, must_have, "movies file")
need(tv, must_have, "tv file")

# Optional columns: create them empty if the file doesn't have them
optional = ["IMDb", "Age", "Rotten Tomatoes", "Genres", "Country", "Language", "Runtime"]
for d in (mv, tv):
    for c in optional:
        if c not in d.columns:
            d[c] = np.nan

mv["type"] = "Movie"
tv["type"] = "TV Show"
print(f"  Raw rows: movies {len(mv):,}, TV {len(tv):,}")

pl = pd.concat([mv, tv], ignore_index=True)
pl = pl.rename(columns={"Title": "title", "Year": "year", "Age": "age",
                        "IMDb": "imdb_orig", "Rotten Tomatoes": "rotten_tomatoes",
                        "Prime Video": "Prime_Video", "Disney+": "Disney_Plus",
                        "Genres": "genres", "Country": "country",
                        "Language": "language", "Runtime": "runtime"})
for c in ["genres", "country", "language", "runtime"]:
    if c not in pl.columns:
        pl[c] = np.nan

pl["title"] = pl["title"].astype(str).str.strip()
pl["title_key"] = pl["title"].map(make_key)
pl["year"] = pd.to_numeric(pl["year"], errors="coerce")
pl["imdb_orig"] = to_number(pl["imdb_orig"])
pl["rotten_tomatoes"] = to_number(pl["rotten_tomatoes"])
for c in ["Netflix", "Hulu", "Prime_Video", "Disney_Plus"]:
    pl[c] = pd.to_numeric(pl[c], errors="coerce").fillna(0).astype(int)

# same title listed twice -> keep one row, combine platform flags
agg = {c: "max" for c in ["Netflix", "Hulu", "Prime_Video", "Disney_Plus"]}
for c in ["title", "age", "imdb_orig", "rotten_tomatoes", "genres",
          "country", "language", "runtime"]:
    agg[c] = "first"
before = len(pl)
pl = pl.groupby(["title_key", "year", "type"], as_index=False, dropna=False).agg(agg)
print(f"  Duplicates merged: {before - len(pl):,}")

pl["is_tv"] = pl["type"].eq("TV Show")
pl = attach_imdb(pl, "year", "is_tv")
pl["imdb_rating"] = pl["imdb_orig"].fillna(pl["imdb_rating_file"])
pl["genres"] = pl["genres"].fillna(pl["imdb_genres"])
pl["runtime"] = pl["runtime"].fillna(pl["imdb_runtime"])
pl["high_rating"] = np.where(pl["imdb_rating"].notna(),
                             (pl["imdb_rating"] >= 7.0).astype(int), np.nan)

names = {"Netflix": "Netflix", "Hulu": "Hulu",
         "Prime_Video": "Prime Video", "Disney_Plus": "Disney+"}
pl["platforms"] = pl.apply(
    lambda r: ", ".join(n for c, n in names.items() if r[c] == 1), axis=1)

pl_cols = ["title", "title_key", "year", "type", "age", "genres", "country",
           "language", "runtime", "Netflix", "Hulu", "Prime_Video", "Disney_Plus",
           "platforms", "rotten_tomatoes", "imdb_id", "imdb_rating", "imdb_votes",
           "high_rating"]
pl[pl_cols].to_csv(CLEAN / "platform_clean.csv", index=False)

print(f"\nPlatform titles: {len(pl):,}  (movies: {(~pl['is_tv']).sum():,}, TV: {pl['is_tv'].sum():,})")
print(f"With an IMDb rating: {pl['imdb_rating'].notna().sum():,} ({pl['imdb_rating'].notna().mean():.1%})")
print(f"With genres: {pl['genres'].notna().sum():,} ({pl['genres'].notna().mean():.1%})")
print("Titles per platform:", {n: int(pl[c].sum()) for c, n in names.items()})
print("\nDone. File saved:", CLEAN / "platform_clean.csv")