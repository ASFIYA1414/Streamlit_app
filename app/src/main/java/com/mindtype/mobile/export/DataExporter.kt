package com.mindtype.mobile.export

import android.content.Context
import android.util.Log
import com.mindtype.mobile.data.AppDatabase

/**
 * Queries all Room tables and returns a CSV string.
 * The Activity is responsible for writing the file via SAF (no permissions needed).
 */
class DataExporter(private val context: Context) {

    suspend fun buildCsvString(): String? {
        return try {
            val db = AppDatabase.getInstance(context)
            val windows = db.featureWindowDao().getAllWindows()
            val labels = db.stressLabelDao().getAllLabels()
            val prefs = context.getSharedPreferences("mindtype_prefs", Context.MODE_PRIVATE)
            val userId = prefs.getString("user_id", "UNKNOWN") ?: "UNKNOWN"

            if (windows.isEmpty()) {
                Log.d("DataExporter", "No data windows found to export")
                return null
            }

            val sb = StringBuilder()
            // Header
            sb.appendLine(
                "user_id,session_id,window_start,window_end," +
                "mean_dwell,std_dwell,mean_flight,std_flight," +
                "typing_speed,backspace_rate,pause_count," +
                "mean_pressure,gyro_std," +
                "raw_score,mapped_class,predicted_class"
            )

            for (w in windows) {
                val matchedLabel = labels.minByOrNull {
                    kotlin.math.abs(it.timestamp - w.windowEnd)
                }?.takeIf { kotlin.math.abs(it.timestamp - w.windowEnd) < 600_000L }

                sb.appendLine(
                    "${userId},${w.sessionId},${w.windowStart},${w.windowEnd}," +
                    "${w.meanDwell},${w.stdDwell},${w.meanFlight},${w.stdFlight}," +
                    "${w.typingSpeed},${w.backspaceRate},${w.pauseCount}," +
                    "${w.meanPressure},${w.gyroStd}," +
                    "${matchedLabel?.rawScore ?: ""},${matchedLabel?.mappedClass ?: ""},${w.predictedClass}"
                )
            }

            sb.toString()
        } catch (e: Exception) {
            Log.e("DataExporter", "CSV build failed: ${e.message}", e)
            null
        }
    }
}
