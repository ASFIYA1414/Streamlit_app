# MindType Mobile 🧠
An Android Edge-AI application that predicts user cognitive stress levels in real-time by analyzing passive keystroke metadata.

## 🚀 Overview
MindType operates as a custom Android keyboard (IME). It transparently intercepts raw keystroke touch events while you type across any app on your phone. To guarantee **absolute user privacy**, the application operates entirely on the "Edge"—meaning no network calls are ever made, and actual typed alphanumeric strings are physically discarded. Only keystroke timing metrics (metadata) are persisted.

These metrics are structured into a 13-feature array and passed through an **on-device ONNX XGBoost Machine Learning Model** that classifies your active cognitive state as *Calm*, *Mild Stress*, or *High Stress*.

## 🧬 Architecture

### 1. Data Collection (`MindTypeIMEService`)
The application intercepts raw Android `MotionEvent` and `KeyEvent` hooks. We extract micro-interactions:
- **Flight Time:** The milliseconds passed between releasing one key and pressing the next.
- **Dwell Time:** The milliseconds a single key is physically held down.
- **Backspace Rate:** The ratio of corrective strokes indicating typing uncertainty.
- **Gyroscope Jitter:** Device movement indicating physical micro-tremors.

### 2. Feature Engineering & Pre-Processing
A sliding temporal window algorithm batches these raw events into discrete 60-second epochs. We calculate standard distribution markers (Mean, Standard Deviation) across the vectors to synthesize exactly 13 floating-point metrics mapping directly to our trained ML pipeline.

### 3. Native Edge-AI Inference (ONNX Runtime)
MindType deploys an off-line `XGBoost` model wrapped in a `.onnx` graph using Android's native `onnxruntime` libraries. Upon receiving the 13-feature batch, the model executes a strict unscaled softmax prediction locally on the phone's CPU. 

### 4. Ground Truth & Validation (WorkManager)
The application periodically hooks into Android's WorkManager API to deliver a system notification every 15 minutes, prompting users to self-report their current stress index (1–5). This forms a paired ground-truth validation set stored asynchronously into an encapsulated SQLite (`Room`) relational database for accuracy calculations and future model tuning.

## 🛠 Tech Stack
* **Language:** Kotlin
* **Machine Learning:** Scikit-Learn -> skl2onnx -> ONNX Runtime Android (Microsoft)
* **Local Persistence:** Android Room Database (SQLite)
* **Background Processing:** Android WorkManager, Coroutine Scopes
* **UI/Data Visuals:** XML, Material Design 3, MPAndroidChart

## 📈 CSV Export & Research Continuity
Via the settings panel, researchers can utilize the Storage Access Framework (SAF) to dump a fully synthesized `data.csv` bundle locally to the Documents directory. This CSV merges computed AI inferences directly against user-submitted Ground Truth values to perform blind evaluation studies.
