package com.mindtype.mobile.ml

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

enum class StressLevel { CALM, STRESSED }

/**
 * Wraps the ONNX model for on-device stress classification.
 * Input: FloatArray[9]
 */
class StressClassifier(private val context: Context) {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    // Scaler constants from Python
    private val means = floatArrayOf(
        73.059f, 31.829f, 519.833f, 826.544f, 136.418f, 0.1575f, 0.5409f, 0.5f, 1981828624.775f
    )
    private val scales = floatArrayOf(
        19.292f, 56.612f, 929.783f, 1509.306f, 82.410f, 0.1821f, 0.7036f, 1.0f, 5323971053.094f
    )

    init {
        try {
            env = OrtEnvironment.getEnvironment()
            val assetManager = context.assets
            val modelBytes = assetManager.open("stress_model.onnx").readBytes()
            session = env?.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
            session = null
            env = null
        }
    }

    fun classify(features: FloatArray): StressLevel {
        val currentEnv = env
        val currentSession = session

        if (currentEnv != null && currentSession != null) {
            try {
                // 1. Standard Scale the input
                val scaledFeatures = FloatArray(9)
                for (i in 0..8) {
                    scaledFeatures[i] = (features[i] - means[i]) / scales[i]
                }

                // 2. Run ONNX Inference
                val inputName = currentSession.inputNames.iterator().next()
                val shape = longArrayOf(1, 9)
                val tensor = OnnxTensor.createTensor(currentEnv, FloatBuffer.wrap(scaledFeatures), shape)

                val result = currentSession.run(Collections.singletonMap(inputName, tensor))
                
                // 3. Extract output
                // ONNX XGBoost output usually contains labels (index 0) and probabilities (index 1)
                // Output 0 is usually Int64 array of predicted classes
                val outputLabels = result[0].value as LongArray
                val prediction = outputLabels[0].toInt()

                tensor.close()
                result.close()

                return if (prediction == 1) StressLevel.STRESSED else StressLevel.CALM
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // --- DYNAMIC DEMO HEURISTIC ALGORITHM ---
        val speedKPM = features[4]
        val backspaceRate = features[5]

        var stressScore = 0
        if (backspaceRate > 0.08f) stressScore += 2
        else if (backspaceRate > 0.04f) stressScore += 1

        if (speedKPM > 140f || (speedKPM < 40f && speedKPM > 5f)) stressScore += 1

        return if (stressScore >= 2) StressLevel.STRESSED else StressLevel.CALM
    }

    fun close() {
        session?.close()
        env?.close()
    }
}
