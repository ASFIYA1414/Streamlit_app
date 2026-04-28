package com.mindtype.mobile.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

enum class StressLevel { CALM, STRESSED }

/**
 * On-device stress classifier using ONNX Runtime Android.
 *
 * Model pipeline:
 *   XGBoost (Python) ──skl2onnx/onnxmltools──▶ mindtype_model.onnx ──▶ OrtSession (Android)
 *
 * Input :  FloatArray[13]  — the 13 behavioral features from FeatureExtractor
 *          Order: mean_dwell, std_dwell, mean_flight, std_flight,
 *                 typing_speed, backspace_rate, pause_count, mean_pressure, gyro_std,
 *                 + 4 extended features (f9–f12, padded with 0 if unused)
 *
 * Output:  StressLevel enum (CALM / STRESSED)
 *
 * If the ONNX model fails to load (e.g. placeholder file) or inference throws,
 * a lightweight heuristic rule-set is used as a fallback so the app keeps running.
 */
class StressClassifier(private val context: Context) {

    companion object {
        const val TAG = "StressClassifier"

        /** Asset filename — copy mindtype_model.onnx from convert_to_onnx.py output */
        const val MODEL_FILE = "mindtype_model.onnx"

        /**
         * Input / output node names produced by skl2onnx with zipmap=False.
         * Run convert_to_onnx.py and read the "[5/5]" section to confirm these.
         * They are almost always "float_input", "label", "probabilities" for sklearn models.
         */
        const val ONNX_INPUT_NAME        = "float_input"
        const val ONNX_OUTPUT_LABEL      = "output_label"      // int64[N]   — predicted class index
        const val ONNX_OUTPUT_PROBS      = "output_probability" // sequence(map) — class probabilities

        const val N_FEATURES = 13 // Updated from 9 to match new XGBoost model
        const val N_CLASSES  = 2
    }

    // OrtEnvironment is a process-wide singleton — do NOT close it prematurely
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    init {
        loadModel()
    }

    // ─── Model Loading ───────────────────────────────────────────────────────

    private fun loadModel() {
        try {
            val modelBytes = context.assets.open(MODEL_FILE).readBytes()
            if (modelBytes.size < 1024) {
                // Placeholder file — too small to be a real ONNX model
                Log.w(TAG, "Model file is only ${modelBytes.size} bytes — placeholder detected. " +
                        "Run convert_to_onnx.py and copy the output to assets/. Heuristic fallback active.")
                ortSession = null
                return
            }

            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortSession = ortEnv.createSession(modelBytes, opts)
            logModelInfo()
            Log.i(TAG, "✅ ONNX model loaded: $MODEL_FILE")

        } catch (e: Exception) {
            Log.w(TAG, "ONNX model load failed — heuristic fallback active. Error: ${e.message}")
            ortSession = null
        }
    }

    private fun logModelInfo() {
        ortSession?.let { session ->
            session.inputInfo.forEach  { (name, info) -> Log.d(TAG, "  Input  '$name': ${info.info}") }
            session.outputInfo.forEach { (name, info) -> Log.d(TAG, "  Output '$name': ${info.info}") }
        }
    }

    // ─── Inference ───────────────────────────────────────────────────────────

    /**
     * Classify a 9-feature window into a stress level.
     * Falls back to heuristic rules if the ONNX model is unavailable or fails.
     */
    fun classify(features: FloatArray): StressLevel {
        // Pad the 9 app features with 0.0f up to the 13 required by the new ONNX model
        val paddedFeatures = FloatArray(N_FEATURES)
        val copyLen = minOf(features.size, N_FEATURES)
        features.copyInto(paddedFeatures, 0, 0, copyLen)

        ortSession?.let { session ->
            try {
                return runOnnxInference(session, paddedFeatures)
            } catch (e: Exception) {
                Log.e(TAG, "ONNX inference failed: ${e.message} — using heuristic fallback")
            }
        }

        return heuristicClassify(features)
    }

    private fun runOnnxInference(session: OrtSession, features: FloatArray): StressLevel {
        // Build input tensor: shape [1, 9]
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(features),
            longArrayOf(1L, N_FEATURES.toLong())
        )

        val result = session.run(mapOf(ONNX_INPUT_NAME to inputTensor))
        inputTensor.close()

        val stressLevel = parseOnnxOutput(result)
        result.close()
        return stressLevel
    }

    /**
     * Parses ONNX Runtime output. Tries the probabilities tensor first (most robust),
     * then falls back to the label (class index) tensor.
     *
     * Both output formats are produced by skl2onnx and onnxmltools — this handles both.
     */
    private fun parseOnnxOutput(result: OrtSession.Result): StressLevel {
        // --- Strategy 1: Use label tensor (most direct & reliable for binary model) ---
        // Model outputs output_label as int64 — 0=CALM, 1=STRESSED
        runCatching {
            val labelValue = result[ONNX_OUTPUT_LABEL].get().value
            val idx: Int = when (labelValue) {
                is LongArray -> labelValue[0].toInt()
                is IntArray  -> labelValue[0]
                is Array<*>  -> (labelValue[0] as Long).toInt()
                else -> throw IllegalStateException("Unexpected label type: ${labelValue?.javaClass}")
            }
            Log.d(TAG, "ONNX label → class index $idx (${if (idx == 0) "CALM" else "STRESSED"})")
            return indexToLevel(idx)
        }.onFailure { Log.w(TAG, "Label parse failed: ${it.message}") }

        // --- Strategy 2: Use probability map (seq(map(int64, float))) ---
        // Output format: [{0: 0.93, 1: 0.07}] — pick argmax
        runCatching {
            val probsRaw = result[ONNX_OUTPUT_PROBS].get().value
            val probMap: Map<*, *> = when (probsRaw) {
                // seq(map) → List of maps, take first
                is List<*> -> probsRaw[0] as Map<*, *>
                is Map<*, *> -> probsRaw
                else -> throw IllegalStateException("Unexpected probs type: ${probsRaw?.javaClass}")
            }
            val prob0 = (probMap[0L] as? Float) ?: (probMap[0] as? Float) ?: 0f
            val prob1 = (probMap[1L] as? Float) ?: (probMap[1] as? Float) ?: 0f
            Log.d(TAG, "ONNX probs → CALM=%.3f STRESSED=%.3f".format(prob0, prob1))
            return indexToLevel(if (prob1 > prob0) 1 else 0)
        }.onFailure { Log.e(TAG, "Probs parse also failed: ${it.message}") }

        throw RuntimeException("Could not parse any ONNX output tensor")
    }

    private fun indexToLevel(index: Int): StressLevel = when (index) {
        0    -> StressLevel.CALM
        else -> StressLevel.STRESSED
    }

    // ─── Heuristic Fallback ──────────────────────────────────────────────────

    /**
     * Rule-based classifier used when the ONNX model is unavailable.
     * Uses the three most discriminative features from the training analysis:
     *   backspace_rate (index 5), gyro_std (index 8), typing_speed (index 4)
     */
    private fun heuristicClassify(features: FloatArray): StressLevel {
        val meanDwell    = features[0]  // ms held per key
        val stdDwell     = features[1]  // variability in hold time
        val meanFlight   = features[2]  // ms between keys
        val speedKPM     = features[4]  // keystrokes per minute
        val backspaceRate = features[5] // fraction of keys that are backspace
        val gyroStd      = features[8]  // phone tilt variability

        var score = 0

        // Backspace rate: normal typing has ~3-6% backspace, stressed = higher
        if      (backspaceRate > 0.12f) score += 3  // very high error rate
        else if (backspaceRate > 0.07f) score += 2  // elevated error rate
        else if (backspaceRate > 0.04f) score += 1  // slightly elevated

        // Physical tremor detected via gyroscope
        if      (gyroStd > 1.5f) score += 2
        else if (gyroStd > 0.8f) score += 1

        // Dwell time variability — stressed = inconsistent key holds
        if (stdDwell > 120f) score += 2
        else if (stdDwell > 60f) score += 1

        // Extreme typing speed deviation from normal (60-100 KPM)
        if (speedKPM > 180f || (speedKPM in 5f..30f)) score += 1

        return when {
            score >= 2 -> StressLevel.STRESSED
            else       -> StressLevel.CALM
        }
    }

    // ─── Lifecycle ──────────────────────────────────────────────────────────

    fun close() {
        ortSession?.close()
        ortSession = null
        // Note: Do NOT close ortEnv — it is a process-wide singleton
    }

    /** True if the real ONNX model is loaded; false means heuristic is active */
    val isModelLoaded: Boolean get() = ortSession != null
}
