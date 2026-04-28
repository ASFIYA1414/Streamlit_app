package com.mindtype.mobile.data.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey

@Entity(
    tableName = "stress_labels",
    foreignKeys = [ForeignKey(
        entity = SessionEntity::class,
        parentColumns = ["session_id"],
        childColumns = ["session_id"],
        onDelete = ForeignKey.CASCADE
    )]
)
data class StressLabelEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "label_id") val labelId: Int = 0,
    @ColumnInfo(name = "session_id") val sessionId: String,
    @ColumnInfo(name = "timestamp") val timestamp: Long = System.currentTimeMillis(),
    /** Raw self-report score 1–5 */
    @ColumnInfo(name = "raw_score") val rawScore: Int,
    /** Mapped class: Calm, Mild_Stress, High_Stress */
    @ColumnInfo(name = "mapped_class") val mappedClass: String
) {
    companion object {
        /**
         * Maps a raw self-report score (1–5) to a class label.
         * Labels are UPPERCASE to match StressLevel.name used in predicted_class
         * (CALM, MILD_STRESS, HIGH_STRESS) — ensures CSV column consistency.
         */
        fun mapScoreToClass(score: Int): String = when (score) {
            1, 2 -> "CALM"
            3    -> "MILD_STRESS"
            else -> "HIGH_STRESS"
        }
    }
}
