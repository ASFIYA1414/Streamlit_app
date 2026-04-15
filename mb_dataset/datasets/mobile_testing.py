import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# -------------------------------------
# LOAD DATA
# -------------------------------------
df = pd.read_csv("clean_mobile_dataset.csv")

# Remove NaNs
df = df.dropna(subset=["mapped_class"])

# Binary conversion
df["mapped_class"] = df["mapped_class"].astype(str).str.strip()
df["mapped_class"] = df["mapped_class"].apply(lambda x: 0 if x == "Calm" else 1)

# -------------------------------------
# DATA AUGMENTATION 🔥
# -------------------------------------
aug = df.copy()

for col in [
    "mean_dwell", "std_dwell", "mean_flight",
    "std_flight", "typing_speed", "backspace_rate"
]:
    noise = np.random.normal(0, 0.08, size=len(aug))
    aug[col] = aug[col] * (1 + noise)

df = pd.concat([df, aug], ignore_index=True)

print("After augmentation:", df.shape)

# -------------------------------------
# CLEAN
# -------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# -------------------------------------
# SIMPLE STRONG FEATURES ONLY 🔥
# -------------------------------------
df["combined_behavior"] = (
    df["mean_dwell"] * df["typing_speed"]
) / (df["backspace_rate"] + 1e-6)

# New Feature
df["dwell_flight_ratio"] = df["mean_dwell"] / (df["mean_flight"] + 1e-6)

X = df[[
    "mean_dwell",
    "std_dwell",
    "mean_flight",
    "std_flight",
    "typing_speed",
    "backspace_rate",
    "combined_behavior",
    "dwell_flight_ratio"
]]

y = df["mapped_class"]

# -------------------------------------
# SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------
# MODEL (SLIGHTLY STRONGER)
# -------------------------------------
# -------------------------------------
# MODEL TUNING (GridSearchCV)
# -------------------------------------
xgb = XGBClassifier(random_state=42, eval_metric="logloss")

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0]
}

print("Running Hyperparameter Tuning (this may take a moment)...")
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\n🔥 Best Hyperparameters found:")
print(grid_search.best_params_)

# -------------------------------------
# PREDICT WITH OPTIMIZED MODEL
# -------------------------------------
# We rely on the natural 0.5 threshold which optimizes for pure accuracy in balanced models
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# -------------------------------------
# RESULTS
# -------------------------------------
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))