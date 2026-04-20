import joblib
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Load your saved pipeline
pipeline = joblib.load("stress_model_pipeline.pkl")

# Extract the two parts separately
scaler = pipeline.named_steps["scaler"]
xgb_model = pipeline.named_steps["xgb"]

# Convert ONLY the XGBoost part to ONNX (9 features)
initial_type = [("float_input", FloatTensorType([None, 9]))]
onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)

# Save
with open("stress_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Save scaler separately — needed in Android to scale input before inference
joblib.dump(scaler, "scaler.pkl")

print("stress_model.onnx saved!")
print("scaler.pkl saved!")
import numpy as np

# Export scaler values for Android hardcoding
print("Scaler means:", scaler.mean_.tolist())
print("Scaler scales:", scaler.scale_.tolist())