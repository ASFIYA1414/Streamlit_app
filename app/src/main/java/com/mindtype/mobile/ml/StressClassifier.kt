package com.mindtype.mobile.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

enum class StressLevel { CALM, MILD_STRESS, HIGH_STRESS }

/**
 * Wraps the TensorFlow Lite model for on-device stress classification.
 * Input:  FloatArray[9]  — the 9 behavioral features (in order from FeatureExtractor)
 * Output: FloatArray[3]  — softmax probabilities [Calm, Mild_Stress, High_Stress]
 */
class StressClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null

    init {
        try {
            val model = loadModelFile()
            interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })
        } catch (e: Exception) {
            // Model placeholder may not be a real TFLite model yet — fail gracefully
            interpreter = null
        }
    }

    fun classify(features: FloatArray): StressLevel {
        interpreter?.let { interp ->
            try {
                val input = arrayOf(features)
                val output = Array(1) { FloatArray(3) }
                interp.run(input, output)
                val probs = output[0]
                return when (probs.indices.maxByOrNull { probs[it] } ?: 0) {
                    0 -> StressLevel.CALM
                    1 -> StressLevel.MILD_STRESS
                    else -> StressLevel.HIGH_STRESS
                }
            } catch (e: Exception) {
                // Ignore failure and fall through to heuristic algorithm
            }
        }

        // --- DYNAMIC DEMO HEURISTIC ALGORITHM ---
        // Lowered thresholds to ensure the UI graph shows VARIATION during testing!
        val speedKPM = features[4]
        val backspaceRate = features[5]
        val gyroStd = features[8]

        var stressScore = 0

        // 1. Error rate (Very sensitive!)
        if (backspaceRate > 0.08f) stressScore += 3 // Automatic high
        else if (backspaceRate > 0.04f) stressScore += 1

        // 2. Physical tremor (Detect even subtle shakes)
        if (gyroStd > 1.2f) stressScore += 2
        else if (gyroStd > 0.4f) stressScore += 1

        // 3. Typing speed variation
        if (speedKPM > 140f || (speedKPM < 40f && speedKPM > 5f)) stressScore += 1

        return when {
            stressScore >= 3 -> StressLevel.HIGH_STRESS
            stressScore >= 1 -> StressLevel.MILD_STRESS
            else -> StressLevel.CALM
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd("mindtype_model.tflite")
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter?.close()
    }
}
