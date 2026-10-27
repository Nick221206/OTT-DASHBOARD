import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "clean" / "platform_clean.csv")

print(df.shape)
print(df[["title", "year", "type", "genres", "imdb_rating", "platforms"]].head(10))
print("\nHigh rating (>=7.0) share:", df["high_rating"].mean().round(3))
print("\nAverage rating by platform:")
for col in ["Netflix", "Hulu", "Prime_Video", "Disney_Plus"]:
    print(f"  {col}: {df.loc[df[col] == 1, 'imdb_rating'].mean():.2f}")