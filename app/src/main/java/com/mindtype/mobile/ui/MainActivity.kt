package com.mindtype.mobile.ui

import android.content.Context
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.mindtype.mobile.R
import com.mindtype.mobile.data.AppDatabase
import com.mindtype.mobile.databinding.ActivityMainBinding
import com.mindtype.mobile.ime.MindTypeIMEService
import com.mindtype.mobile.ml.StressLevel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var db: AppDatabase

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        db = AppDatabase.getInstance(applicationContext)

        binding.btnSettings.setOnClickListener {
            startActivity(android.content.Intent(this, SettingsActivity::class.java))
        }

        // Refresh dashboard every 30 seconds
        lifecycleScope.launch {
            while (isActive) {
                loadDashboard()
                delay(30_000)
            }
        }
    }

    private suspend fun loadDashboard() {
        val prefs = getSharedPreferences("mindtype_prefs", Context.MODE_PRIVATE)
        val sessionId = prefs.getString("current_session_id", "") ?: ""
        val userId = prefs.getString("user_id", "—") ?: "—"

        val since24h = System.currentTimeMillis() - 24 * 60 * 60 * 1000L

        val keystrokeCount = withContext(Dispatchers.IO) {
            db.keystrokeEventDao().countSince(since24h)
        }
        val windows = withContext(Dispatchers.IO) {
            db.featureWindowDao().getWindowsSince(since24h)
        }
        val sessionEntity = withContext(Dispatchers.IO) {
            db.sessionDao().getSessionsForUser(userId).firstOrNull()
        }

        val sessionDurationMin = sessionEntity?.let {
            val endMs = it.endTime ?: System.currentTimeMillis()
            (endMs - it.startTime) / 60_000L
        } ?: 0L

        val avgStress = windows.mapNotNull { w ->
            when (w.predictedClass) {
                "CALM" -> 1f
                "STRESSED" -> 2f
                else -> null
            }
        }.average().let { if (it.isNaN()) 0.0 else it }

        val currentLevel = MindTypeIMEService.currentStressLevel

        withContext(Dispatchers.Main) {
            binding.tvUserId.text = "PARTICIPANT: $userId"
            binding.tvKeystrokeCount.text = "$keystrokeCount"
            binding.tvSessionDuration.text = "${sessionDurationMin}m"
            binding.tvAvgStress.text = "%.1f".format(avgStress)

            // Current stress dot
            val (color, label) = when (currentLevel) {
                StressLevel.CALM -> Pair(getColor(R.color.stress_calm), "🟢 Calm")
                StressLevel.STRESSED -> Pair(getColor(R.color.stress_high), "🔴 Stressed")
            }
            binding.tvCurrentStress.text = label
            binding.tvCurrentStress.setTextColor(color)

            // Build stress trend chart
            if (windows.isNotEmpty()) {
                val entries = windows.mapIndexed { i, w ->
                    val y = when (w.predictedClass) {
                        "CALM" -> 1f
                        "STRESSED" -> 2f
                        else -> 0f
                    }
                    Entry(i.toFloat(), y)
                }
                val dataSet = LineDataSet(entries, "Behavioral Trend").apply {
                    setColor(getColor(R.color.primary_accent))
                    setCircleColor(getColor(R.color.primary_accent))
                    lineWidth = 3f
                    circleRadius = 4f
                    setDrawCircles(true)
                    setDrawCircleHole(false)
                    setDrawFilled(true)
                    fillColor = getColor(R.color.primary_accent)
                    fillAlpha = 20
                    setDrawValues(false)
                    mode = LineDataSet.Mode.CUBIC_BEZIER
                }
                
                binding.stressChart.apply {
                    data = LineData(dataSet)
                    description.isEnabled = false
                    legend.isEnabled = false
                    xAxis.isEnabled = false
                    axisLeft.isEnabled = false
                    axisRight.isEnabled = false
                    setDrawGridBackground(false)
                    setTouchEnabled(false)
                    invalidate()
                }
            }
        }
    }
}
