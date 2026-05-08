package com.speedometer

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.atan2

class SensorTracker(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gravitySensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)

    @Volatile
    private var currentPitchRad: Float = 0f

    private val alpha = 0.2f

    fun startListening() {
        gravitySensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun stopListening() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_GRAVITY) {
            val y = event.values[1]
            val z = event.values[2]

            val rawPitch = atan2(z.toDouble(), y.toDouble()).toFloat()

            currentPitchRad = if (currentPitchRad == 0f) {
                rawPitch
            } else {
                currentPitchRad + alpha * (rawPitch - currentPitchRad)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {

    }

    fun getCurrentPitchRad(): Float {
        return currentPitchRad
    }
}