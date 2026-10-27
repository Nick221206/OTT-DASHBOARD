import pandas as pd
from pathlib import Path

# Finds the OTT_Project folder automatically
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
print("Looking in:", RAW)

if not RAW.exists():
    print("The folder data/raw does not exist. Create it and put your files inside.")
    raise SystemExit

print("\nFiles found:")
for f in sorted(RAW.iterdir()):
    print("  ", f.name, f"({f.stat().st_size / 1e6:.1f} MB)")

def peek(name, **kwargs):
    path = RAW / name
    if not path.exists():
        print(f"\n!! {name} NOT FOUND")
        return
    df = pd.read_csv(path, nrows=20000, **kwargs)
    print(f"\n=== {name} ===")
    print("Columns:", list(df.columns))

peek("netflix_titles.csv")
peek("MoviesOnStreamingPlatforms_updated.csv")
peek("tv_shows.csv")
peek("title.ratings.tsv.gz", sep="\t")
peek("title.basics.tsv.gz", sep="\t", na_values="\\N", low_memory=False)