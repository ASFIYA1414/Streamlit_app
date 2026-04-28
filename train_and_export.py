"""
MindType — Complete Train → Test → Export Pipeline
====================================================
Binary classification: 0 = CALM, 1 = STRESSED

Usage:
    python train_and_export.py --data path/to/your_dataset.csv

The script will:
  1. Load and clean your CSV dataset
  2. Train an XGBoost binary classifier
  3. Print accuracy, F1 score, and confusion matrix
  4. Export to mindtype_model.onnx
  5. Copy the .onnx to app/src/main/assets/

Requirements (already installed in venv):
    xgboost scikit-learn pandas numpy onnxruntime skl2onnx onnxmltools joblib imbalanced-learn
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train + Export MindType binary stress model")
parser.add_argument("--data", type=str, required=True,
                    help="Path to your dataset CSV file")
parser.add_argument("--output", type=str, default="mindtype_model.onnx",
                    help="Output ONNX model path (default: mindtype_model.onnx)")
parser.add_argument("--label-col", type=str, default=None,
                    help="Name of the label column in your CSV (auto-detected if not set)")
parser.add_argument("--threshold", type=float, default=2.0,
                    help="If label is numeric, values >= threshold → STRESSED (default: 2.0)")
args = parser.parse_args()

ASSETS_DIR = os.path.join(os.path.dirname(__file__),
                           "app", "src", "main", "assets")

# ─── Feature columns (must match FeatureExtractor.kt output) ─────────────────

FEATURE_COLS = [
    "mean_dwell", "std_dwell", "mean_flight", "std_flight",
    "typing_speed", "backspace_rate", "pause_count",
    "mean_pressure", "gyro_std",
    # Extended features (set to 0 if not in your CSV — will be padded)
    "f9", "f10", "f11", "f12"
]

N_FEATURES = 13

# ─── Step 1: Load data ───────────────────────────────────────────────────────

print(f"\n[1/5] Loading dataset from: {args.data}")
df = pd.read_csv(args.data)
print(f"      Shape: {df.shape}")
print(f"      Columns: {list(df.columns)}")

# ─── Step 2: Detect label column ─────────────────────────────────────────────

print(f"\n[2/5] Detecting label column...")

label_col = args.label_col
if label_col is None:
    # Auto-detect: look for common label column names
    candidates = ["label", "stress_label", "predicted_class", "mapped_class",
                  "stress", "stress_level", "class"]
    for c in candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        print(f"      ERROR: Could not auto-detect label column.")
        print(f"      Available columns: {list(df.columns)}")
        print(f"      Use --label-col <column_name> to specify it.")
        sys.exit(1)

print(f"      Using label column: '{label_col}'")
print(f"      Unique values: {df[label_col].unique()}")

# ─── Step 3: Encode labels to binary ─────────────────────────────────────────

print(f"\n[3/5] Encoding labels to binary (0=CALM, 1=STRESSED)...")

raw_labels = df[label_col]

# If labels are already numeric
if pd.api.types.is_numeric_dtype(raw_labels):
    y = (raw_labels >= args.threshold).astype(int)
    print(f"      Numeric labels: values >= {args.threshold} → STRESSED")

# If labels are strings like CALM / MILD_STRESS / HIGH_STRESS / STRESSED
elif pd.api.types.is_string_dtype(raw_labels) or pd.api.types.is_object_dtype(raw_labels):
    def encode_label(v):
        v = str(v).strip().upper()
        if v in ("CALM", "0", "NOT_STRESSED"):
            return 0
        elif v in ("STRESSED", "1", "MILD_STRESS", "HIGH_STRESS", "2", "3"):
            return 1
        else:
            return -1  # unknown

    y = raw_labels.apply(encode_label)
    unknown = (y == -1).sum()
    if unknown > 0:
        print(f"      WARNING: {unknown} rows with unknown labels dropped.")
        mask = y != -1
        df = df[mask]
        y = y[mask]
    print(f"      String labels encoded.")
else:
    print(f"      ERROR: Unsupported label type: {raw_labels.dtype}")
    sys.exit(1)

print(f"      Class distribution:")
print(f"        CALM (0)     : {(y==0).sum()} rows")
print(f"        STRESSED (1) : {(y==1).sum()} rows")

# ─── Step 4: Build feature matrix ────────────────────────────────────────────

print(f"\n[4/5] Building feature matrix ({N_FEATURES} features)...")

available_features = []
for col in FEATURE_COLS:
    if col in df.columns:
        available_features.append(col)
    else:
        # Add as zero column
        df[col] = 0.0
        available_features.append(col)

X = df[available_features].fillna(0.0).values.astype(np.float32)
print(f"      Feature matrix shape: {X.shape}")

# ─── Step 5: Train/test split ────────────────────────────────────────────────

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

# ─── Step 6: Scale features ───────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Step 7: Train XGBoost ───────────────────────────────────────────────────

from xgboost import XGBClassifier

print(f"\n      Training XGBoost binary classifier...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_s, y_train)

# ─── Step 8: Evaluate ────────────────────────────────────────────────────────

y_pred = model.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")
cm  = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  ✅ RESULTS")
print(f"{'='*50}")
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  F1 Score : {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {cm}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["CALM", "STRESSED"]))
print(f"{'='*50}\n")

# ─── Step 9: Save model ───────────────────────────────────────────────────────

joblib.dump({"model": model, "scaler": scaler}, "mindtype_model.pkl")
print(f"      Saved: mindtype_model.pkl")

# ─── Step 10: Export to ONNX ─────────────────────────────────────────────────

print(f"[5/5] Exporting to ONNX...")

from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as convert_xgb_to_onnx

# Rename features to f0, f1, ... for ONNX compatibility
booster = model.get_booster()
booster.feature_names = [f"f{i}" for i in range(N_FEATURES)]

# Wrap scaler + model in a Pipeline for single-step ONNX export
pipeline = Pipeline([
    ("scaler", scaler),
    ("classifier", model)
])

update_registered_converter(
    XGBClassifier, "XGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgb_to_onnx,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]}
)

initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset={"": 17, "ai.onnx.ml": 3}
)

with open(args.output, "wb") as f:
    f.write(onnx_model.SerializeToString())

size_kb = os.path.getsize(args.output) / 1024
print(f"      Saved ONNX: {args.output} ({size_kb:.1f} KB)")

# ─── Step 11: Validate ONNX ──────────────────────────────────────────────────

import onnxruntime as ort

sess = ort.InferenceSession(args.output)
inp  = sess.get_inputs()[0].name
outs = [o.name for o in sess.get_outputs()]
print(f"      Input  : '{inp}'")
for i, o in enumerate(sess.get_outputs()):
    print(f"      Output[{i}]: '{o.name}'")

# Quick sanity check
calm_sample    = np.array([[120, 15, 95, 45, 0.02, 0.1, 0.5, 0.5, 0.0, 0, 0, 0, 0]], dtype=np.float32)
stressed_sample = np.array([[70, 50, 40, 145, 0.15, 0.3, 1.2, 0.8, 0.3, 0, 0, 0, 0]], dtype=np.float32)

for name, sample in [("CALM sample", calm_sample), ("STRESSED sample", stressed_sample)]:
    result = sess.run(outs, {inp: sample})
    label = int(np.array(result[0]).flatten()[0])
    print(f"      {name} → Predicted: {'CALM' if label == 0 else 'STRESSED'}")

# ─── Step 12: Copy to Android assets ────────────────────────────────────────

os.makedirs(ASSETS_DIR, exist_ok=True)
dest = os.path.join(ASSETS_DIR, "mindtype_model.onnx")
import shutil
shutil.copy2(args.output, dest)
print(f"\n  ✅ Copied ONNX model to Android assets: {dest}")
print(f"  🚀 Now rebuild your APK in Android Studio!")
