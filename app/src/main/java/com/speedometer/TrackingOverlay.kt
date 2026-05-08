package com.speedometer

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View


class TrackingOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // ── Paints ────────────────────────────────────────────────────
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color       = Color.parseColor("#00E5FF")
        style       = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CC000000")
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.parseColor("#00E5FF")
        textSize = 38f
        typeface = Typeface.MONOSPACE
        style    = Paint.Style.FILL
    }

    // ── State ─────────────────────────────────────────────────────
    private var vehicleData: FloatArray = FloatArray(0)

    /** Raw sensor dimensions */
    private var sensorW = 1f
    private var sensorH = 1f

    // ── Public API ────────────────────────────────────────────────

    fun setFrameSize(w: Int, h: Int) {
        sensorW = w.toFloat()
        sensorH = h.toFloat()
    }

    fun update(data: FloatArray) {
        vehicleData = data
        postInvalidate()
    }

    // ── Drawing ───────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (vehicleData.isEmpty()) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        val visualW = sensorH
        val visualH = sensorW

        val scaleX = viewW / visualW
        val scaleY = viewH / visualH

        val count = vehicleData.size / 6
        for (i in 0 until count) {
            val id     = vehicleData[i * 6 + 0].toInt()
            // Sensor-space box
            val sx     = vehicleData[i * 6 + 1]
            val sy     = vehicleData[i * 6 + 2]
            val sw     = vehicleData[i * 6 + 3]
            val sh     = vehicleData[i * 6 + 4]
            val speed  = vehicleData[i * 6 + 5]

            // Centre in sensor space
            val scx = sx + sw / 2f
            val scy = sy + sh / 2f

            val dcx = (sensorH - scy) * scaleX
            val dcy = scx * scaleY

            // Box size also swaps axes
            val dw  = sh * scaleX
            val dh  = sw * scaleY

            val left   = dcx - dw / 2f
            val top    = dcy - dh / 2f
            val right  = dcx + dw / 2f
            val bottom = dcy + dh / 2f

            // ── Draw box ─────────────────────────────────────────
            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = if (speed >= 1f) "${speed.toInt()} km/h" else "tracking"
            val textW = textPaint.measureText(label)
            val textH = textPaint.textSize
            val labelPad = 6f

            val labelLeft   = left
            val labelTop    = (top - textH - labelPad * 2).coerceAtLeast(0f)
            val labelRight  = left + textW + labelPad * 2
            val labelBottom = labelTop + textH + labelPad * 2

            canvas.drawRect(labelLeft, labelTop, labelRight, labelBottom, labelBgPaint)
            canvas.drawText(label, labelLeft + labelPad, labelBottom - labelPad, textPaint)

            // Soo Corner markers for better visual precision
            drawCornerMarkers(canvas, left, top, right, bottom)
        }
    }

    /**
     * Draws small L-shaped corner markers inside the bounding box corners.
     * This makes it much easier to judge alignment accuracy visually.
     */
    private fun drawCornerMarkers(
        canvas: Canvas,
        l: Float, t: Float, r: Float, b: Float
    ) {
        val arm = minOf((r - l) * 0.20f, (b - t) * 0.20f, 40f)
        with(canvas) {
            // Top-left
            drawLine(l, t, l + arm, t, boxPaint)
            drawLine(l, t, l, t + arm, boxPaint)
            // Top-right
            drawLine(r - arm, t, r, t, boxPaint)
            drawLine(r, t, r, t + arm, boxPaint)
            // Bottom-left
            drawLine(l, b - arm, l, b, boxPaint)
            drawLine(l, b, l + arm, b, boxPaint)
            // Bottom-right
            drawLine(r - arm, b, r, b, boxPaint)
            drawLine(r, b - arm, r, b, boxPaint)
        }
    }
}