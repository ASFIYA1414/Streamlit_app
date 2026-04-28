# Add project specific ProGuard rules here.

# ── ONNX Runtime Android ───────────────────────────────────────────────────
# OrtSession, OnnxTensor, OrtEnvironment use JNI — must not be renamed/removed
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# ── Room Entities ──────────────────────────────────────────────────────────
-keep class com.mindtype.mobile.data.entity.** { *; }

# ── WorkManager Workers ────────────────────────────────────────────────────
-keep class com.mindtype.mobile.workers.** { *; }

# ── ML Classifier (accessed from service) ─────────────────────────────────
-keep class com.mindtype.mobile.ml.** { *; }
