package com.mindtype.mobile.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey

@Entity(
    tableName = "feature_windows",
    foreignKeys = [ForeignKey(
        entity = SessionEntity::class,
        parentColumns = ["session_id"],
        childColumns = ["session_id"],
        onDelete = ForeignKey.CASCADE
    )]
)
data class FeatureWindowEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "window_id") val windowId: Int = 0,
    @ColumnInfo(name = "session_id") val sessionId: String = "",
    @ColumnInfo(name = "window_start") val windowStart: Long,
    @ColumnInfo(name = "window_end") val windowEnd: Long,

    // 9 behavioral features
    @ColumnInfo(name = "mean_dwell") val meanDwell: Float,
    @ColumnInfo(name = "std_dwell") val stdDwell: Float,
    @ColumnInfo(name = "mean_flight") val meanFlight: Float,
    @ColumnInfo(name = "std_flight") val stdFlight: Float,
    @ColumnInfo(name = "typing_speed") val typingSpeed: Float,
    @ColumnInfo(name = "backspace_rate") val backspaceRate: Float,
    @ColumnInfo(name = "pause_count") val pauseCount: Int,
    @ColumnInfo(name = "mean_pressure") val meanPressure: Float,
    @ColumnInfo(name = "gyro_std") val gyroStd: Float,
    @ColumnInfo(name = "combined_behavior") val combinedBehavior: Float = 0f,

    // ML output
    @ColumnInfo(name = "predicted_class") val predictedClass: String = "",
    // Self-reported class (nullable — may not exist for every window)
    @ColumnInfo(name = "stress_label") val stressLabel: String? = null
) {
    /** Returns the 9 features in the exact order the TFLite model expects */
    fun toFeatureArray(): FloatArray = floatArrayOf(
        meanDwell, stdDwell, meanFlight, stdFlight,
        typingSpeed, backspaceRate, pauseCount.toFloat(),
        meanPressure, combinedBehavior
    )
}
