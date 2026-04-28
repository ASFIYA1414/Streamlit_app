package com.mindtype.mobile.ime

import android.content.Context
import android.graphics.*
import android.graphics.drawable.Drawable
import android.inputmethodservice.Keyboard
import android.inputmethodservice.KeyboardView
import android.util.AttributeSet
import androidx.core.content.ContextCompat
import com.mindtype.mobile.R

/**
 * Premium custom KeyboardView with modern rounded-key rendering,
 * distinct accent/special key styling, and smooth press feedback.
 */
class MindTypeKeyboardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : KeyboardView(context, attrs) {

    // ── Paint for key labels ─────────────────────────────────────────────────
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private val specialLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#C7C7CC")
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif", Typeface.NORMAL)
    }

    private val accentLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
    }

    // ── Drawables (lazy loaded) ──────────────────────────────────────────────
    private val keyBgNormal: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_normal)!! }
    private val keyBgPressed: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_pressed)!! }
    private val specialBgNormal: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_special_normal)!! }
    private val specialBgPressed: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_special_pressed)!! }
    private val accentBgNormal: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_accent_normal)!! }
    private val accentBgPressed: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_accent_pressed)!! }
    private val spaceBgNormal: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_space_normal)!! }
    private val spaceBgPressed: Drawable by lazy { ContextCompat.getDrawable(context, R.drawable.key_bg_space_pressed)!! }

    // Key inset (gap between logical key rect and painted rounded-rect)
    private val keyInset = (3 * resources.displayMetrics.density).toInt()

    // Shadow/elevation paint for subtle key depth
    private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#18000000")
        maskFilter = BlurMaskFilter(2f * resources.displayMetrics.density, BlurMaskFilter.Blur.NORMAL)
    }

    override fun onDraw(canvas: Canvas) {
        // Don't call super — we draw everything ourselves
        val keyboard = keyboard ?: return
        val keys = keyboard.keys

        // Draw keyboard background
        canvas.drawColor(Color.parseColor("#1C1C1E"))

        for (key in keys) {
            val isPressed = key.pressed
            val isSpecial = isSpecialKey(key)
            val isAccent = isAccentKey(key)
            val isSpace = key.codes.firstOrNull() == 32

            // Compute inset key bounds
            val left = key.x + keyInset
            val top = key.y + keyInset
            val right = key.x + key.width - keyInset
            val bottom = key.y + key.height - keyInset

            // Pick the right drawable
            val bg = when {
                isSpace && isPressed -> spaceBgPressed
                isSpace -> spaceBgNormal
                isAccent && isPressed -> accentBgPressed
                isAccent -> accentBgNormal
                isSpecial && isPressed -> specialBgPressed
                isSpecial -> specialBgNormal
                isPressed -> keyBgPressed
                else -> keyBgNormal
            }

            // Draw subtle shadow under key (not for pressed state)
            if (!isPressed) {
                val shadowRect = RectF(
                    left.toFloat() + 1f,
                    top.toFloat() + 2f,
                    right.toFloat() + 1f,
                    bottom.toFloat() + 2f
                )
                canvas.drawRoundRect(shadowRect, 8f * resources.displayMetrics.density,
                    8f * resources.displayMetrics.density, shadowPaint)
            }

            // Draw key background
            bg.setBounds(left, top, right, bottom)
            bg.draw(canvas)

            // Draw label
            val label = key.label?.toString() ?: ""
            if (label.isNotEmpty()) {
                val paint = when {
                    isAccent -> accentLabelPaint
                    isSpecial -> specialLabelPaint
                    else -> labelPaint
                }
                // Size depends on key type
                paint.textSize = when {
                    isSpace -> 13f * resources.displayMetrics.scaledDensity
                    isSpecial -> 14f * resources.displayMetrics.scaledDensity
                    label.length > 1 -> 13f * resources.displayMetrics.scaledDensity
                    else -> 20f * resources.displayMetrics.scaledDensity
                }

                val centerX = (left + right) / 2f
                val centerY = (top + bottom) / 2f - (paint.descent() + paint.ascent()) / 2f
                canvas.drawText(label, centerX, centerY, paint)
            }
        }
    }

    private fun isSpecialKey(key: Keyboard.Key): Boolean {
        val code = key.codes.firstOrNull() ?: return false
        return code < 0 && code != -12 // all negative codes except search (-12) are special
                || code == 32 // space is drawn separately but logically special
    }

    private fun isAccentKey(key: Keyboard.Key): Boolean {
        val code = key.codes.firstOrNull() ?: return false
        return code == -12 // search button gets accent styling
    }
}
