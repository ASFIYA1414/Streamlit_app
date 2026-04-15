import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("clean_mobile_dataset.csv")

print("Dataset shape (before fix):", df.shape)

# -------------------------------
# FIX LABEL INCONSISTENCY (CRITICAL)
# -------------------------------
df["mapped_class"] = df["mapped_class"].astype(str).str.lower().str.strip()

df["mapped_class"] = df["mapped_class"].replace({
    "calm": "Calm",
    "high_stress": "High_Stress",
    "mild_stress": "Mild_Stress"
})

# -------------------------------
# 🔥 FINAL SAFETY CLEAN
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)

# Fill labels using predicted_class if needed
if "predicted_class" in df.columns:
    df["mapped_class"] = df["mapped_class"].fillna(df["predicted_class"])

# Fill numeric NaNs
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop only if label missing
df = df.dropna(subset=["mapped_class"])

print("Dataset shape (after fix):", df.shape)
print("\nClass distribution:\n", df["mapped_class"].value_counts())

# -------------------------------
# 🔥 FEATURE ENGINEERING
# -------------------------------
df["combined_behavior"] = (
    df["mean_dwell"] * df["typing_speed"]
) / (df["backspace_rate"] + 1e-6)

# -------------------------------
# FEATURES & LABEL
# -------------------------------
X = df[[
    "mean_dwell",
    "std_dwell",
    "mean_flight",
    "std_flight",
    "typing_speed",
    "backspace_rate",
    "pause_count",
    "mean_pressure",
    "combined_behavior"
]]

y = df["mapped_class"]

# -------------------------------
# CHECK NaNs
# -------------------------------
print("\nNaNs in X:\n", pd.DataFrame(X).isna().sum())
print("\nNaNs in y:\n", pd.Series(y).isna().sum())

# -------------------------------
# MODEL PIPELINE
# -------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# -------------------------------
# HYPERPARAMETER TUNING
# -------------------------------
param_grid = {
    "rf__n_estimators": [50, 100],
    "rf__max_depth": [3, 5, 7],
    "rf__min_samples_split": [5, 10],
    "rf__min_samples_leaf": [2, 4]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=skf,
    scoring="f1_weighted",
    n_jobs=-1
)

# -------------------------------
# TRAIN
# -------------------------------
grid.fit(X, y)

print("\nBest Parameters:", grid.best_params_)

# -------------------------------
# FINAL EVALUATION
# -------------------------------
accuracy = cross_val_score(grid.best_estimator_, X, y, cv=skf, scoring="accuracy")
f1 = cross_val_score(grid.best_estimator_, X, y, cv=skf, scoring="f1_weighted")

print("\nFINAL RESULTS:")
print("Accuracy:", np.mean(accuracy))
print("F1-score:", np.mean(f1))