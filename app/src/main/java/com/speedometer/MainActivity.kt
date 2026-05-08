package com.speedometer

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // Variables
    private lateinit var viewFinder: PreviewView
    private lateinit var speedText: TextView
    private lateinit var statusText: TextView
    private lateinit var maxSpeedText: TextView
    private lateinit var pitchText: TextView
    private lateinit var actionButton: TextView
    private lateinit var trackingOverlay: TrackingOverlay
    private lateinit var sensorTracker: SensorTracker
    private lateinit var cameraExecutor: ExecutorService

    private var maxSpeed = 0f

    private var isTrackingActive = false
    private var lockedPitchRad = 0f

    // Requesting Permission
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) startCamera() else finish()
    }


    private external fun initTracker(onnxPath: String)

    private fun initAIEngine() {
          val modelPath = copyAssetToStorage("yolo26n.onnx")

        if (modelPath != null) {
            initTracker(modelPath)
            Log.d("speedometer", "YOLOv4-tiny loaded successfully!")
        } else {
            Log.e("speedometer", "Failed to load AI assets")
        }
    }

    // copy files from asset folder
    private fun copyAssetToStorage(filename: String): String? {
        val file = java.io.File(filesDir, filename)
        if (file.exists()) return file.absolutePath

        return try {
            assets.open(filename).use { inputStream ->
                java.io.FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // Runs on App Creation
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        trackingOverlay = findViewById(R.id.trackingOverlay)
        viewFinder = findViewById(R.id.viewFinder)
        speedText = findViewById(R.id.speedText)
        statusText = findViewById(R.id.statusText)
        maxSpeedText = findViewById(R.id.maxSpeedText)
        pitchText = findViewById(R.id.pitchText)
        actionButton = findViewById(R.id.actionButton)

        cameraExecutor = Executors.newSingleThreadExecutor()
        sensorTracker = SensorTracker(this)

        // Update Text
        actionButton.setOnClickListener {
            isTrackingActive = !isTrackingActive

            if (isTrackingActive) {
                lockedPitchRad = sensorTracker.getCurrentPitchRad()

                val pitchDeg = Math.toDegrees(lockedPitchRad.toDouble()).toFloat()
                pitchText.text = String.format("%.1f°", pitchDeg)

                actionButton.text = "◼ STOP TRACKING"
                actionButton.setBackgroundColor(android.graphics.Color.parseColor("#FF2D55"))
                actionButton.setTextColor(android.graphics.Color.WHITE)

                statusText.text = "● SCANNING"
                statusText.setTextColor(android.graphics.Color.parseColor("#00E5FF"))
            } else {
                actionButton.text = "▶ START TRACKING"
                actionButton.setBackgroundColor(android.graphics.Color.parseColor("#00E5FF"))
                actionButton.setTextColor(android.graphics.Color.parseColor("#040810"))

                statusText.text = "◼ IDLE"
                statusText.setTextColor(android.graphics.Color.parseColor("#8000E5FF"))
                speedText.text = "000"
            }
        }

        // Run AI Model
        initAIEngine()

        if (allPermissionsGranted()) startCamera() else requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // Resumes Tracking
    override fun onResume() {
        super.onResume()
        sensorTracker.startListening()
    }

    // On Pause of Tracking
    override fun onPause() {
        super.onPause()
        sensorTracker.stopListening()
    }

    // On App Shutdown
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    // Starting CameraX
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!isTrackingActive) {
                            image.close()
                            return@setAnalyzer
                        }

                        trackingOverlay.setFrameSize(image.width, image.height)

                        val yPlane = image.planes[0]

                        val results = processNativeFrame(
                            yPlane.buffer, image.width, image.height, yPlane.rowStride,
                            lockedPitchRad, image.imageInfo.timestamp
                        )

                        if (results != null && results.isNotEmpty()) {
                            Log.d("Speedometer", "AI DETECTED ${results.size / 6} OBJECTS")
                        } else {
                            Log.v("Speedometer", "AI is scanning but found Nothing")
                        }

                        // Update UI
                        runOnUiThread {
                            trackingOverlay.update(results ?: FloatArray(0))

                            if (results != null && results.isNotEmpty()) {
                                val firstCarSpeed = results[5]
                                if (firstCarSpeed > 0) {
                                    updateSpeedUI(firstCarSpeed)
                                }
                            }
                        }
                        image.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch (exc: Exception) { Log.e("Speedometer", "Use case binding failed", exc) }

        }, ContextCompat.getMainExecutor(this))
    }

    // Update Detected Speed Display
    private fun updateSpeedUI(speedKmh: Float) {
        speedText.text = String.format("%03d", speedKmh.toInt())

        statusText.text = "TRACKING"
        statusText.setTextColor(android.graphics.Color.parseColor("#FF2D55"))

        statusText.postDelayed({
            if (isTrackingActive) {
                statusText.text = "SCANNING"
                statusText.setTextColor(android.graphics.Color.parseColor("#00E5FF"))
            }
        }, 1000)

        if (speedKmh > maxSpeed) {
            maxSpeed = speedKmh
            maxSpeedText.text = maxSpeed.toInt().toString()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    // JNI link
    private external fun processNativeFrame(yBuffer: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, pitchRad: Float, timestampNs: Long): FloatArray?

    companion object {
        init { System.loadLibrary("speedometer") }
    }

}