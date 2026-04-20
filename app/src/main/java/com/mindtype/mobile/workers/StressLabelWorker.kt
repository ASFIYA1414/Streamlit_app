package com.mindtype.mobile.workers

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.core.app.NotificationCompat
import androidx.work.*
import com.mindtype.mobile.R
import com.mindtype.mobile.data.AppDatabase
import com.mindtype.mobile.data.entity.StressLabelEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

/**
 * WorkManager worker that fires every 10 minutes to prompt the user for a self-reported stress score.
 * Shows a dialog via notification action, auto-dismisses after 30 seconds.
 */
class StressLabelWorker(
    private val context: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(context, workerParams) {

    companion object {
        const val WORK_NAME = "stress_label_periodic"
        const val NOTIF_ID = 2001
        const val CHANNEL_ID = "stress_prompt_channel"

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<StressLabelWorker>(15, TimeUnit.MINUTES)
                .setConstraints(Constraints.Builder().build())
                .build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.Main) {
        createChannel()

        val prefs = context.getSharedPreferences("mindtype_prefs", Context.MODE_PRIVATE)
        val sessionId = prefs.getString("current_session_id", "") ?: ""

        // Build notification with 5 inline action buttons (1–5)
        val nm = context.getSystemService(NotificationManager::class.java)

        val builder = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("How stressed do you feel right now?")
            .setContentText("Tap a number: 1 = Calm  ·  5 = Very Stressed")
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setTimeoutAfter(30_000) // auto-dismiss after 30s

        val options = listOf(
            Triple(1, "Calm", "1"),
            Triple(3, "Mild", "3"),
            Triple(5, "High", "5")
        )

        for ((score, label, _) in options) {
            val intent = StressResponseReceiver.buildIntent(context, score, sessionId)
            val pi = android.app.PendingIntent.getBroadcast(
                context, score, intent, android.app.PendingIntent.FLAG_IMMUTABLE or android.app.PendingIntent.FLAG_UPDATE_CURRENT
            )
            builder.addAction(0, label, pi)
        }

        nm.notify(NOTIF_ID, builder.build())
        Result.success()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Stress Self-Report Prompts", NotificationManager.IMPORTANCE_HIGH
            ).apply { description = "Periodic stress check-in prompts" }
            context.getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }
}
