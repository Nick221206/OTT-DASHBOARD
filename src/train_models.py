import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FIG = ROOT / "figures"
MODELS.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "clean" / "platform_clean.csv")
df = df.dropna(subset=["high_rating", "year"]).copy()
df["high_rating"] = df["high_rating"].astype(int)
print(f"Titles with a rating: {len(df):,}")
print(f"High-rating share (>=7.0): {df['high_rating'].mean():.1%}")

# ---------- features ----------
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["runtime"] = df["runtime"].fillna(df.groupby("type")["runtime"].transform("median"))
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["age"] = df["age"].fillna("Unknown").astype(str)
df["is_tv"] = (df["type"] == "TV Show").astype(int)

# multi-hot genres (top 25)
gl = df["genres"].fillna("").str.split(",").apply(lambda xs: [x.strip() for x in xs if x.strip()])
top_genres = pd.Series([g for xs in gl for g in xs]).value_counts().head(25).index
G = pd.DataFrame({f"genre_{g}": gl.apply(lambda xs, g=g: int(g in xs)) for g in top_genres},
                 index=df.index)

age = pd.get_dummies(df["age"], prefix="age").astype(int)
plat = df[["Netflix", "Hulu", "Prime_Video", "Disney_Plus"]]
X = pd.concat([df[["year", "runtime", "is_tv"]], plat, G, age], axis=1)
y = df["high_rating"]
print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- models ----------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=42),
}

baseline = max(y_test.mean(), 1 - y_test.mean())
rows, fitted = [], {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC_AUC": roc_auc_score(y_test, prob)})
    fitted[name] = m
    print(f"\n{name} confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, pred))

res = pd.DataFrame(rows).round(3)
print(f"\nBaseline (always predict the majority class): {baseline:.3f}")
print("\n", res.to_string(index=False))
res.to_csv(MODELS / "model_comparison.csv", index=False)

# ---------- save ----------
best = res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
print("\nBest model by ROC-AUC:", best)
joblib.dump(fitted["Random Forest"], MODELS / "random_forest.joblib")
joblib.dump(fitted["XGBoost"], MODELS / "xgboost.joblib")
joblib.dump(fitted[best], MODELS / "best_model.joblib")
joblib.dump(list(X.columns), MODELS / "feature_columns.joblib")

# ---------- feature importance of best model ----------
imp = pd.Series(fitted[best].feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, color="#4C72B0")
plt.title(f"Top 15 features: {best}")
plt.tight_layout()
plt.savefig(FIG / "09_feature_importance.png", dpi=150)
plt.close()
print("Saved figures/09_feature_importance.png")
print("Models saved in:", MODELS)