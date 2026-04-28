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
import com.mindtype.mobile.ml.StressClassifier
import com.mindtype.mobile.ml.StressLevel
import androidx.core.content.ContextCompat
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

        // Show ONNX model deployment status immediately on launch
        val classifier = StressClassifier(applicationContext)
        val modelLoaded = classifier.isModelLoaded
        classifier.close()
        binding.tvModelStatus.text = if (modelLoaded)
            "🟢 ONNX Model Active"
        else
            "🟡 Heuristic Mode — deploy mindtype_model.onnx to activate ML"
        binding.tvModelStatus.setTextColor(
            if (modelLoaded) getColor(R.color.stress_calm) else getColor(R.color.stress_mild)
        )

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

        val avgStress = if (windows.isEmpty()) 0.0 else
            (windows.count { it.predictedClass == "CALM" }.toDouble() / windows.size) * 100.0

        val currentLevel = MindTypeIMEService.currentStressLevel

        withContext(Dispatchers.Main) {
            binding.tvUserId.text = "PARTICIPANT: $userId"
            binding.tvKeystrokeCount.text = "$keystrokeCount"
            binding.tvSessionDuration.text = "${sessionDurationMin}m"
            binding.tvAvgStress.text = if (windows.isEmpty()) "—" else "${avgStress.toInt()}%"

            // Current stress dot
            val (color, label) = when (currentLevel) {
                StressLevel.CALM     -> Pair(getColor(R.color.stress_calm), "🟢 Calm")
                StressLevel.STRESSED -> Pair(getColor(R.color.stress_high), "🔴 Stressed")
            }
            binding.tvCurrentStress.text = label
            binding.tvCurrentStress.setTextColor(color)

            // ── Stress Trend Chart ───────────────────────────────────────────
            if (windows.isNotEmpty()) {
                val calmColor   = getColor(R.color.stress_calm)
                val stressColor = getColor(R.color.stress_high)
                val accentColor = getColor(R.color.primary_accent)

                val entries = windows.mapIndexed { i, w ->
                    Entry(i.toFloat(), if (w.predictedClass == "STRESSED") 2f else 1f)
                }

                val dotColors = windows.map { w ->
                    if (w.predictedClass == "STRESSED") stressColor else calmColor
                }

                val dataSet = LineDataSet(entries, "Stress").apply {
                    setColor(accentColor)
                    lineWidth = 2.5f
                    mode = LineDataSet.Mode.STEPPED
                    setDrawCircles(true)
                    circleRadius = 5f
                    circleHoleRadius = 2.5f
                    circleColors = dotColors
                    circleHoleColor = getColor(R.color.primary_surface)
                    setDrawValues(false)
                    setDrawFilled(true)
                    fillDrawable = ContextCompat.getDrawable(
                        this@MainActivity, R.drawable.chart_gradient)
                    highLightColor = android.graphics.Color.TRANSPARENT // no crosshair
                    isHighlightEnabled = false
                }

                binding.stressChart.apply {
                    data = LineData(dataSet)
                    description.isEnabled = false
                    legend.isEnabled = false
                    setNoDataText("")
                    setDrawGridBackground(false)
                    setTouchEnabled(false) // disable touch completely — cleaner look
                    setExtraOffsets(4f, 16f, 16f, 4f)

                    // X-Axis — clean, minimal
                    xAxis.apply {
                        position = com.github.mikephil.charting.components.XAxis
                            .XAxisPosition.BOTTOM
                        setDrawGridLines(false)
                        setDrawAxisLine(false)
                        textColor = getColor(R.color.text_disabled)
                        textSize = 9f
                        granularity = 1f
                        setLabelCount(minOf(entries.size, 5), false)
                        valueFormatter = object :
                            com.github.mikephil.charting.formatter.ValueFormatter() {
                            override fun getFormattedValue(value: Float): String =
                                "#${value.toInt() + 1}"
                        }
                    }

                    // Y-Axis — just 2 clean labels
                    axisLeft.apply {
                        setDrawAxisLine(false)
                        gridColor = getColor(R.color.divider_dark)
                        gridLineWidth = 1f
                        enableGridDashedLine(6f, 6f, 0f)
                        textColor = getColor(R.color.text_medium_emphasis)
                        textSize = 10f
                        setLabelCount(2, true)
                        axisMinimum = 0.5f
                        axisMaximum = 2.5f
                        valueFormatter = object :
                            com.github.mikephil.charting.formatter.ValueFormatter() {
                            override fun getFormattedValue(value: Float): String =
                                when (value.toInt()) {
                                    1 -> "CALM"
                                    2 -> "STRESSED"
                                    else -> ""
                                }
                        }
                    }
                    axisRight.isEnabled = false

                    animateY(800, com.github.mikephil.charting.animation.Easing.EaseOutCubic)
                    invalidate()
                }

            } else {
                binding.stressChart.apply {
                    clear()
                    setNoDataText("Type with MindType keyboard to see your stress trend")
                    setNoDataTextColor(getColor(R.color.text_disabled))
                    getPaint(com.github.mikephil.charting.charts.Chart.PAINT_INFO).textSize =
                        resources.displayMetrics.scaledDensity * 11f
                    invalidate()
                }
            }

        }
    }

}
