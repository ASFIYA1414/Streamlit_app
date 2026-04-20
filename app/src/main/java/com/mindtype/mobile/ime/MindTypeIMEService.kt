package com.mindtype.mobile.ime

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorManager
import android.inputmethodservice.InputMethodService
import android.inputmethodservice.Keyboard
import android.inputmethodservice.KeyboardView
import android.os.Build
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import androidx.core.app.NotificationCompat
import com.mindtype.mobile.R
import com.mindtype.mobile.data.AppDatabase
import com.mindtype.mobile.data.entity.KeystrokeEventEntity
import com.mindtype.mobile.features.FeatureExtractor
import com.mindtype.mobile.ml.StressClassifier
import com.mindtype.mobile.ml.StressLevel
import com.mindtype.mobile.ui.MainActivity
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

enum class KeyboardMode { ALPHA, NUMBERS, SYMBOLS }

class MindTypeIMEService : InputMethodService(), KeyboardView.OnKeyboardActionListener {

    companion object {
        const val TAG = "MindTypeIME"
        const val NOTIFICATION_ID = 1001
        const val CHANNEL_ID = "mindtype_stress_channel"
        var currentStressLevel: StressLevel = StressLevel.CALM

        // Custom mode-switch codes
        const val CODE_SWITCH_NUMBERS  = -2
        const val CODE_SWITCH_ALPHA    = -10
        const val CODE_SWITCH_SYMBOLS  = -11
        const val CODE_SHIFT           = -1
    }

    private lateinit var keyboardView: KeyboardView
    private lateinit var gyroscopeManager: GyroscopeManager
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var stressClassifier: StressClassifier
    private lateinit var db: AppDatabase

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private val keyPressTimes = mutableMapOf<Int, Long>()
    private var lastEventTime: Long = 0L
    private var sessionId: String = ""
    private var userId: String = ""

    // Keyboard state
    private var currentMode = KeyboardMode.ALPHA
    private var isShifted = false

    override fun onCreate() {
        super.onCreate()
        db = AppDatabase.getInstance(applicationContext)
        featureExtractor = FeatureExtractor()
        stressClassifier = StressClassifier(applicationContext)
        gyroscopeManager = GyroscopeManager(
            getSystemService(Context.SENSOR_SERVICE) as SensorManager
        )
        gyroscopeManager.start()

        val prefs = getSharedPreferences("mindtype_prefs", Context.MODE_PRIVATE)
        userId = prefs.getString("user_id", "U00") ?: "U00"
        sessionId = prefs.getString("current_session_id", "") ?: ""

        Log.d(TAG, "onCreate: IME service started, userId=$userId")
        createNotificationChannel()
        updateNotification(StressLevel.CALM)
    }

    override fun onCreateInputView(): View {
        keyboardView = layoutInflater.inflate(R.layout.keyboard_view, null) as KeyboardView
        keyboardView.isPreviewEnabled = false
        keyboardView.setOnKeyboardActionListener(this)
        Log.d(TAG, "onCreateInputView: keyboard view inflated, listener set")
        loadKeyboard(KeyboardMode.ALPHA)
        return keyboardView
    }

    private fun loadKeyboard(mode: KeyboardMode) {
        currentMode = mode
        val xmlRes = when (mode) {
            KeyboardMode.ALPHA   -> R.xml.keys_layout
            KeyboardMode.NUMBERS -> R.xml.keys_numbers
            KeyboardMode.SYMBOLS -> R.xml.keys_symbols
        }
        val kb = Keyboard(this, xmlRes)
        if (mode == KeyboardMode.ALPHA) kb.isShifted = isShifted
        keyboardView.keyboard = kb
        keyboardView.invalidateAllKeys()
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        featureExtractor.resetWindow()
        lastEventTime = 0L
        // Return to alpha on new input field
        if (::keyboardView.isInitialized) {
            loadKeyboard(KeyboardMode.ALPHA)
        }
    }

    // ─── Key handling ─────────────────────────────────────────────────────────

    override fun onPress(primaryCode: Int) {
        keyPressTimes[primaryCode] = System.currentTimeMillis()
    }

    override fun onRelease(primaryCode: Int) {
        keyPressTimes.remove(primaryCode)
    }

    override fun onKey(primaryCode: Int, keyCodes: IntArray?) {
        val eventTime = System.currentTimeMillis()
        val downTime = keyPressTimes[primaryCode] ?: (eventTime - 70L)

        // Handle mode-switch and utility keys
        when (primaryCode) {
            CODE_SWITCH_NUMBERS -> { Log.d(TAG, "→ Mode: NUMBERS"); loadKeyboard(KeyboardMode.NUMBERS); return }
            CODE_SWITCH_ALPHA   -> { Log.d(TAG, "→ Mode: ALPHA");   isShifted = false; loadKeyboard(KeyboardMode.ALPHA); return }
            CODE_SWITCH_SYMBOLS -> { Log.d(TAG, "→ Mode: SYMBOLS"); loadKeyboard(KeyboardMode.SYMBOLS); return }
            CODE_SHIFT -> {
                isShifted = !isShifted
                Log.d(TAG, "Shift toggled: isShifted=$isShifted")
                keyboardView.keyboard?.isShifted = isShifted
                keyboardView.invalidateAllKeys()
                return
            }
        }

        Log.d(TAG, "onKey: code=$primaryCode dwell=${eventTime-downTime}ms")

        // Record the event (no text stored)
        processKeyEvent(
            keyCode = primaryCode,
            downTime = downTime,
            eventTime = eventTime,
            pressure = 0.5f,
            touchSize = 0.5f,
            isBackspace = (primaryCode == Keyboard.KEYCODE_DELETE)
        )

        // Emit character to the connected app
        val ic = currentInputConnection ?: return
        when (primaryCode) {
            Keyboard.KEYCODE_DELETE -> ic.deleteSurroundingText(1, 0)
            Keyboard.KEYCODE_DONE  -> ic.performEditorAction(EditorInfo.IME_ACTION_DONE)
            else -> {
                val char = when {
                    isShifted && primaryCode in 97..122 -> primaryCode.toChar().uppercaseChar().toString()
                    else -> primaryCode.toChar().toString()
                }
                ic.commitText(char, 1)
                // Auto-cancel shift after one letter
                if (isShifted && primaryCode in 97..122) {
                    isShifted = false
                    keyboardView.keyboard?.isShifted = false
                    keyboardView.invalidateAllKeys()
                }
            }
        }
    }

    fun processKeyEvent(
        keyCode: Int,
        downTime: Long,
        eventTime: Long,
        pressure: Float,
        touchSize: Float,
        isBackspace: Boolean
    ) {
        val dwellTime = (eventTime - downTime).toFloat()
        val flightTime = if (lastEventTime > 0L) (downTime - lastEventTime).toFloat() else 0f
        lastEventTime = eventTime

        val event = KeystrokeEventEntity(
            sessionId = sessionId,
            userId = userId,
            timestamp = System.currentTimeMillis(),
            keyCode = keyCode,
            downTime = downTime,
            eventTime = eventTime,
            dwellTime = dwellTime,
            flightTime = flightTime,
            touchPressure = pressure,
            touchSize = touchSize,
            isBackspace = if (isBackspace) 1 else 0
        )

        serviceScope.launch { db.keystrokeEventDao().insert(event) }

        val gyroReadings = gyroscopeManager.getRecentReadings()
        featureExtractor.addEvent(event, gyroReadings)

        featureExtractor.getWindowIfReady(gyroReadings)?.let { window ->
            gyroscopeManager.clearReadings()
            serviceScope.launch {
                val windowEntity = window.copy(sessionId = sessionId)
                val stressLevel = stressClassifier.classify(window.toFeatureArray())
                val updatedWindow = windowEntity.copy(predictedClass = stressLevel.name)
                db.featureWindowDao().insert(updatedWindow)
                currentStressLevel = stressLevel
                updateNotification(stressLevel)
            }
        }
    }

    // ─── Notification ─────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "MindType Stress Monitor",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Shows your current stress indicator" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(level: StressLevel): Notification {
        val (emoji, label) = when (level) {
            StressLevel.CALM -> Pair("🟢", "Calm")
            StressLevel.STRESSED -> Pair("🔴", "Stressed")
        }
        val intent = Intent(this, MainActivity::class.java)
        val pi = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("MindType $emoji")
            .setContentText("Stress: $label — tap to view dashboard")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(pi)
            .build()
    }

    private fun updateNotification(level: StressLevel) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(level))
    }

    // ─── Unused listener stubs ────────────────────────────────────────────────

    override fun onText(text: CharSequence?) {}
    override fun swipeLeft() {}
    override fun swipeRight() {}
    override fun swipeDown() {}
    override fun swipeUp() {}

    override fun onDestroy() {
        super.onDestroy()
        gyroscopeManager.stop()
        stressClassifier.close()
        serviceJob.cancel()
    }
}
