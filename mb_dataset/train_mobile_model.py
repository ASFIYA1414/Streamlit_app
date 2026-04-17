import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
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
df = df[df["mapped_class"] != "nan"]

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

y_raw = df["mapped_class"]

# XGBoost requires numerical classes (0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(y_raw)

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
    ("xgb", XGBClassifier(
        random_state=42,
        eval_metric='mlogloss'
    ))
])

# -------------------------------
# HYPERPARAMETER TUNING
# -------------------------------
param_grid = {
    "xgb__n_estimators": [100, 200, 300],
    "xgb__max_depth": [5, 7, 9],
    "xgb__learning_rate": [0.01, 0.05, 0.1]
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