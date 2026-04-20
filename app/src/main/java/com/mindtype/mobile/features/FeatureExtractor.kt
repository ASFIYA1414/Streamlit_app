package com.mindtype.mobile.features

import com.mindtype.mobile.data.entity.KeystrokeEventEntity
import com.mindtype.mobile.data.entity.FeatureWindowEntity
import kotlin.math.sqrt

/**
 * Computes the 9 behavioral features from a 60-second sliding window of keystroke events.
 *
 * Features:
 *  1. mean_dwell      — mean key hold duration (ms)
 *  2. std_dwell       — standard deviation of dwell time
 *  3. mean_flight     — mean inter-key gap (ms)
 *  4. std_flight      — standard deviation of flight time
 *  5. typing_speed    — keys per minute
 *  6. backspace_rate  — ratio of KEYCODE_DEL events to total
 *  7. pause_count     — inter-key gaps > 2000ms
 *  8. mean_pressure   — mean touch pressure (0.0–1.0)
 *  9. gyro_std        — std dev of gyroscope Z-axis readings
 */
class FeatureExtractor {

    private val WINDOW_DURATION_MS = 15_000L // 15 seconds (reduced for fast UI demonstration)
    private val PAUSE_THRESHOLD_MS = 2_000f

    private val events = mutableListOf<KeystrokeEventEntity>()
    private var windowStart: Long = System.currentTimeMillis()

    fun resetWindow() {
        events.clear()
        windowStart = System.currentTimeMillis()
    }

    fun addEvent(event: KeystrokeEventEntity, gyroReadings: List<Float>) {
        events.add(event)
    }

    /**
     * Returns a FeatureWindowEntity if the 60-second window has elapsed and there are
     * enough events, otherwise returns null. Resets the window after returning.
     */
    fun getWindowIfReady(gyroReadings: List<Float> = emptyList()): FeatureWindowEntity? {
        val now = System.currentTimeMillis()
        if (now - windowStart < WINDOW_DURATION_MS) return null
        if (events.size < 5) {
            windowStart = now
            events.clear()
            return null
        }

        val windowEnd = now
        val snapshot = events.toList()
        val oldStart = windowStart
        windowStart = now
        events.clear()

        return buildFeatureWindow(snapshot, oldStart, windowEnd, gyroReadings)
    }

    private fun buildFeatureWindow(
        snapshot: List<KeystrokeEventEntity>,
        start: Long,
        end: Long,
        gyroReadings: List<Float>
    ): FeatureWindowEntity {

        val dwells = snapshot.map { it.dwellTime }
        val flights = snapshot.filter { it.flightTime > 0f }.map { it.flightTime }
        val pressures = snapshot.map { it.touchPressure }

        val meanDwell = dwells.average().toFloat()
        val stdDwell = stdDev(dwells)
        val meanFlight = if (flights.isEmpty()) 0f else flights.average().toFloat()
        val stdFlight = stdDev(flights)

        val windowDurationMin = (end - start) / 60_000.0
        val typingSpeed = (snapshot.size / windowDurationMin).toFloat()

        val backspaceCount = snapshot.count { it.isBackspace == 1 }
        val backspaceRate = backspaceCount.toFloat() / snapshot.size.toFloat()

        val pauseCount = flights.count { it > PAUSE_THRESHOLD_MS }

        val meanPressure = if (pressures.isEmpty()) 0f else pressures.average().toFloat()

        val gyroStd = stdDev(gyroReadings)

        val combinedBehavior = (meanDwell * typingSpeed) / (backspaceRate + 1e-6f)

        return FeatureWindowEntity(
            windowStart = start,
            windowEnd = end,
            meanDwell = meanDwell,
            stdDwell = stdDwell,
            meanFlight = meanFlight,
            stdFlight = stdFlight,
            typingSpeed = typingSpeed,
            backspaceRate = backspaceRate,
            pauseCount = pauseCount,
            meanPressure = meanPressure,
            gyroStd = gyroStd,
            combinedBehavior = combinedBehavior
        )
    }

    private fun stdDev(values: List<Float>): Float {
        if (values.size < 2) return 0f
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        return sqrt(variance).toFloat()
    }
}
