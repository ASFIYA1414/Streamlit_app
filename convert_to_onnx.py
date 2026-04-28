"""
MindType — XGBoost → ONNX Conversion Script
============================================
Run this script ONCE on your desktop/laptop (where the trained model lives)
to produce mindtype_model.onnx, then copy it into the Android app.

Requirements:
    pip install onnxmltools skl2onnx onnxruntime joblib xgboost numpy

Usage:
    python convert_to_onnx.py --model path/to/mindtype_model.pkl

Output:
    mindtype_model.onnx  (copy this to app/src/main/assets/)
"""

import argparse
import os
import sys
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Convert MindType XGBoost model to ONNX")
parser.add_argument(
    "--model",
    type=str,
    default="mindtype_model.pkl",
    help="Path to the trained XGBoost model file (.pkl, .json, or .ubj)"
)
parser.add_argument(
    "--output",
    type=str,
    default="mindtype_model.onnx",
    help="Output .onnx file path"
)
parser.add_argument(
    "--scaler",
    type=str,
    default=None,
    help="Optional: path to a fitted sklearn StandardScaler/MinMaxScaler (.pkl)"
)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load the model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading model from: {args.model}")

import xgboost as xgb

model_ext = os.path.splitext(args.model)[1].lower()

if model_ext == ".pkl":
    import joblib
    loaded_obj = joblib.load(args.model)
    
    # If the file is a dictionary, try to find the model inside it
    if isinstance(loaded_obj, dict):
        print(f"      Loaded object is a dictionary. Searching for model...")
        model = None
        # Try common keys
        for key in ["model", "clf", "classifier", "estimator", "rf", "xgb"]:
            if key in loaded_obj:
                model = loaded_obj[key]
                print(f"      Found model in key: '{key}'")
                break
        
        # If not found by key, try searching values for something with a predict method
        if model is None:
            for k, v in loaded_obj.items():
                if hasattr(v, "predict") or hasattr(v, "get_booster"):
                    model = v
                    print(f"      Found model in key: '{k}'")
                    break
        
        if model is None:
            print(f"ERROR: Could not find a model object inside the dictionary keys: {list(loaded_obj.keys())}")
            sys.exit(1)
    else:
        model = loaded_obj

    print(f"      Loaded model type: {type(model).__name__}")
elif model_ext in (".json", ".ubj"):
    # Native XGBoost format — wrap in sklearn-style Booster
    booster = xgb.Booster()
    booster.load_model(args.model)
    model = booster
    print(f"      Loaded native XGBoost Booster")
else:
    print(f"ERROR: Unsupported model format '{model_ext}'. Use .pkl or .json")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Convert to ONNX
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[2/5] Converting to ONNX (9 input features → 3 output classes)...")

# If the model is a pipeline (imblearn or sklearn), strip out training-only steps like SMOTE
if "Pipeline" in type(model).__name__:
    print(f"      Pipeline detected. Stripping training-only steps (SMOTE, etc.)...")
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    inference_steps = []
    for name, step in model.steps:
        step_type = type(step).__name__.lower()
        if any(x in step_type for x in ["smote", "over_sample", "under_sample", "nearmiss"]):
            print(f"      - Skipping training-only step: '{name}' ({type(step).__name__})")
            continue
        inference_steps.append((name, step))
        print(f"      + Keeping inference step: '{name}' ({type(step).__name__})")
    
    model = SklearnPipeline(inference_steps)

# If an external scaler was provided via --scaler, wrap it
if args.scaler:
    print(f"      External scaler provided: {args.scaler} — embedding into ONNX graph...")
    import joblib
    from sklearn.pipeline import Pipeline as SklearnPipeline
    scaler_obj = joblib.load(args.scaler)
    
    if hasattr(model, "predict_proba"):
        # Wrap existing model (which might be a Pipe or XGBClassifier) in a new Pipe with the scaler
        model = SklearnPipeline([("external_scaler", scaler_obj), ("classifier", model)])
        print(f"      Wrapped model with external scaler.")
    else:
        print("      WARNING: Native Booster detected — external scaler cannot be embedded.")

# --- ONNX conversion STRICTLY requires feature names to be f0, f1, f2... ---
# If trained with pandas DataFrames, the names will be strings. We strip them here.
print(f"      Ensuring all XGBoost feature names follow 'f%d' pattern for ONNX...")
import xgboost as xgb
def rename_booster_features(obj):
    if hasattr(obj, "get_booster"):
        try:
            b = obj.get_booster()
            b.feature_names = [f"f{i}" for i in range(len(b.feature_names))]
            print(f"      - Renamed features in {type(obj).__name__}")
        except Exception as e:
            pass
    elif hasattr(obj, "steps"):
        for _, step in obj.steps:
            rename_booster_features(step)

rename_booster_features(model)

from skl2onnx.common.data_types import FloatTensorType

N_FEATURES = 13  # Must match your training feature count
N_CLASSES   = 2  # Binary: CALM (0), STRESSED (1)

try:
    # Path A: sklearn-wrapped XGBoost (XGBClassifier or Pipeline) — preferred
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as convert_xgb_to_onnx
    from xgboost import XGBClassifier
    
    # Register XGBClassifier so skl2onnx knows how to handle it inside a pipeline
    update_registered_converter(
        XGBClassifier, 'XGBClassifier',
        calculate_linear_classifier_output_shapes, convert_xgb_to_onnx,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
    )
    print("      Registered XGBClassifier converter with skl2onnx.")

    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    
    # We will remove options={'zipmap': False} because it gets passed to the registered XGBClassifier
    # component which might reject it. By default, skl2onnx outputs a sequence/dict of probabilities,
    # and our Android code handles both formats (flat array or dict/sequence return).
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={'': 17, 'ai.onnx.ml': 3}
    )
    print("      Conversion path: skl2onnx (sklearn-compatible XGBClassifier)")

except Exception as e1:
    import traceback
    print(f"      skl2onnx path failed ({e1})")
    print(traceback.format_exc())
    print("      trying onnxmltools...")
    try:
        # Path B: onnxmltools (native Booster or older XGBoost)
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OtFloatTensorType
        initial_type = [("float_input", OtFloatTensorType([None, N_FEATURES]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=17)
        print("      Conversion path: onnxmltools")
    except Exception as e2:
        print(f"ERROR: Both conversion paths failed.\n  skl2onnx: {e1}\n  onnxmltools: {e2}")
        print("  Make sure you installed: pip install onnxmltools skl2onnx")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Save to disk
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[3/5] Saving ONNX model to: {args.output}")
with open(args.output, "wb") as f:
    f.write(onnx_model.SerializeToString())

size_kb = os.path.getsize(args.output) / 1024
print(f"      File size: {size_kb:.1f} KB")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Validate with ONNX Runtime
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Validating model with ONNX Runtime...")

import onnxruntime as ort

sess = ort.InferenceSession(args.output)
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

print(f"      Input  : '{input_name}' shape={sess.get_inputs()[0].shape} dtype={sess.get_inputs()[0].type}")
for i, o in enumerate(sess.get_outputs()):
    print(f"      Output[{i}]: '{o.name}' shape={o.shape} dtype={o.type}")

# Run a test inference using 13 features matching the model
test_cases = {
    "Calm (slow, steady typing)":
        np.array([[120.0, 15.0, 95.0, 45.0, 0.02, 0.1, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32),
    "Stressed (fast + errors + shaky)":
        np.array([[70.0, 50.0, 40.0, 145.0, 0.15, 0.3, 1.2, 0.8, 0.3, 2.5, 2.0, 0.5, 2.0]], dtype=np.float32),
}

CLASS_NAMES = ["CALM", "STRESSED"]

print("\n      Test Inference Results:")
print("      " + "-" * 60)
all_ok = True
for scenario, features in test_cases.items():
    result = sess.run(output_names, {input_name: features})
    # result[0] = label (int or array), result[1] = probabilities (sequence of maps)
    if len(result) >= 2:
        # Extract the dictionary of probabilities from the sequence
        prob_dict = result[1][0] if isinstance(result[1], list) else result[1]
        
        if isinstance(prob_dict, dict):
            # Sort the dictionary values by key (0, 1, 2)
            prob_list = [prob_dict.get(k, 0.0) for k in sorted(prob_dict.keys())]
            probs = np.array(prob_list)
        else:
            probs = np.array(prob_dict).flatten()
            
        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx] if predicted_idx < 3 else "UNKNOWN"
        print(f"      [{scenario}]")
        print(f"        → Predicted: {predicted_class} | Probs: {probs.round(3).tolist()}")
    else:
        label = int(np.array(result[0]).flatten()[0])
        predicted_class = CLASS_NAMES[label] if label < 3 else "UNKNOWN"
        print(f"      [{scenario}]")
        print(f"        → Predicted: {predicted_class}")

print("      " + "-" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Print output names (needed for Android StressClassifier.kt)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[5/5] ✅ ONNX model is ready!")
print(f"\n      IMPORTANT — note these values for your Android app:")
print(f"      Input name  : \"{input_name}\"")
for i, name in enumerate(output_names):
    print(f"      Output[{i}] name: \"{name}\"")

print(f"""
──────────────────────────────────────────────────────────────────────────────
 NEXT STEPS:
 1. Copy the file into your Android project:
    cp {args.output} /Users/meghanaepari/dev/mindtype_mobileapp/app/src/main/assets/mindtype_model.onnx

 2. Update build.gradle — the dependency + aaptOptions are already updated.

 3. The Android StressClassifier.kt has been rewritten to use ONNX Runtime.
    Make sure ONNX_INPUT_NAME and ONNX_OUTPUT_PROBS_NAME match the values above.
──────────────────────────────────────────────────────────────────────────────
""")
