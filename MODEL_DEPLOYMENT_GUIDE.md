# MindType Model Deployment Guide
## XGBoost → ONNX → Android

> This guide tells you exactly how to get the trained model (86–87% accuracy)
> running inside the Android app. Follow the steps in order.

---

## Prerequisites

Have these ready:
- Your trained model file (`mindtype_model.pkl` or `mindtype_model.json`)
- Python 3.8+ environment with XGBoost installed
- This repository cloned on your Mac

---

## Step 1 — Install Python conversion dependencies

```bash
pip install onnxmltools skl2onnx onnxruntime joblib xgboost numpy
```

---

## Step 2 — Run the conversion script

From the root of the repository:

```bash
python convert_to_onnx.py \
    --model /path/to/mindtype_model.pkl \
    --output mindtype_model.onnx
```

**Expected output (tail):**
```
[4/5] Validating model with ONNX Runtime...
      Input  : 'float_input' shape=[None, 9]  dtype=tensor(float)
      Output[0]: 'label'         shape=[None]    dtype=tensor(int64)
      Output[1]: 'probabilities' shape=[None, 3] dtype=tensor(float)

      Test Inference Results:
      ------------------------------------------------------------
      [Calm (slow, steady typing)]
        → Predicted: CALM | Probs: [0.923, 0.061, 0.016]
      [Mild Stress (faster + some errors)]
        → Predicted: MILD_STRESS | Probs: [0.112, 0.764, 0.124]
      [High Stress (very fast/erratic)]
        → Predicted: HIGH_STRESS | Probs: [0.031, 0.189, 0.780]

[5/5] ✅ ONNX model is ready!
      IMPORTANT — note these values for your Android app:
      Input name  : "float_input"
      Output[0] name: "label"
      Output[1] name: "probabilities"
```

> **If the input/output names differ from above**, update the constants in
> `StressClassifier.kt`:
> ```kotlin
> const val ONNX_INPUT_NAME   = "float_input"   // ← change if different
> const val ONNX_OUTPUT_LABEL = "label"          // ← change if different
> const val ONNX_OUTPUT_PROBS = "probabilities"  // ← change if different
> ```

---

## Step 3 — Copy the model into Android assets

```bash
cp mindtype_model.onnx \
   app/src/main/assets/mindtype_model.onnx
```

Verify the file is large enough (should be > 100 KB for a real XGBoost model):
```bash
ls -lh app/src/main/assets/
# mindtype_model.onnx  →  should be ~150–500 KB
# mindtype_model.tflite → old placeholder (108 bytes), can be deleted
```

---

## Step 4 — Build and run

```bash
./gradlew assembleDebug
# or use Android Studio → Run
```

On first launch of the app, Logcat will show:
```
D/StressClassifier: ✅ ONNX model loaded: mindtype_model.onnx
D/StressClassifier:   Input  'float_input': ...
D/StressClassifier:   Output 'label': ...
D/StressClassifier:   Output 'probabilities': ...
```

The dashboard will update to show:
```
🟢 ONNX Model Active
```

---

## Feature Order (must match training!)

The classifier sends features to the model in **exactly this order**:

| Index | Feature | Unit |
|---|---|---|
| 0 | `mean_dwell` | ms |
| 1 | `std_dwell` | ms |
| 2 | `mean_flight` | ms |
| 3 | `std_flight` | ms |
| 4 | `typing_speed` | keys/min |
| 5 | `backspace_rate` | ratio 0–1 |
| 6 | `pause_count` | count |
| 7 | `mean_pressure` | 0.0–1.0 |
| 8 | `gyro_std` | rad/s std dev |

> This is defined in `FeatureWindowEntity.toFeatureArray()`. Make sure your
> training script used **the same feature order and scaling**.

---

## If You Used a Scaler (MinMaxScaler / StandardScaler)

If your training pipeline included a scaler **before** the XGBoost classifier,
pass it to the conversion script:

```bash
python convert_to_onnx.py \
    --model mindtype_model.pkl \
    --scaler mindtype_scaler.pkl \
    --output mindtype_model.onnx
```

The scaler will be embedded into the ONNX graph as a preprocessing step.
The Android app sends **raw feature values** — the ONNX model handles scaling internally.

> If no `--scaler` flag is passed, the script assumes features are already
> normalized before being passed to XGBoost. Confirm this matches your training.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model file is only 108 bytes` in Logcat | Placeholder still in assets | Run Steps 2–3 |
| `ONNX inference failed: No entry for key 'float_input'` | Input name mismatch | Update `ONNX_INPUT_NAME` constant |
| All predictions are `CALM` | Scaler missing / wrong feature order | Verify feature order and scaler |
| App crashes on startup | ONNX Runtime not in build | Run `./gradlew assembleDebug` after build.gradle change |
| Logcat: `ONNX model load failed` | Corrupted asset file | Re-copy the `.onnx` file |

---

## Verifying Model Is Active

1. Open the app → Main Dashboard
2. Look at the hero card — it shows either:
   - 🟢 **ONNX Model Active** — real model running
   - 🟡 **Heuristic Mode** — placeholder, deploy model
3. In Logcat, filter by `StressClassifier` to see per-window inference logs

---

*MindType Mobile v1.1 | VIT-AP University | April 2026*
