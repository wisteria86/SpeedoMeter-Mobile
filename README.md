# SpeedoMeter Mobile

Project-Members: 23L-0553, 23L-0718.

A high-performance Android application that detects and calculates the speed of moving vehicles and pedestrians in real-time. 

Built with a hybrid **Kotlin + C++** architecture, this app leverages the power of native C++ to run complex neural networks and physics algorithms smoothly on mobile devices, ensuring high frame rates and accurate tracking.

## Key Features
* **Real-Time Object Detection:** Uses a lightweight YOLO nano model (`yolo26n`) to identify cars, buses, trucks, and people instantly.
* **Accurate Speed Tracking:** Projects 2D pixel coordinates into 3D real-world space using camera focal length and pitch angles to calculate true ground displacement.
* **Multi-Object Tracking:** Assigns unique IDs to multiple vehicles simultaneously without confusing them, even in busy traffic.
* **Smooth UI Overlays:** Draws stable bounding boxes and speed labels directly over the live camera preview.

## Under the Hood (The Algorithms)
This app following computer vision pipeline:
* **Deep Neural Networks (DNN):** Runs YOLO natively via OpenCV C++ for maximum mobile efficiency.
* **Kalman Filters (6-State):** Acts as a physics engine to predict where a vehicle will be in the next frame. This smooths out camera jitters and keeps bounding boxes perfectly sized as vehicles approach.
* **Ground Distance Computation:** Uses trigonometric pinhole camera modeling to translate 2D screen pixels into actual physical meters on the road.
* **Exponential Moving Average (EMA):** Filters out microscopic tracking noise to provide a clean, stable speed reading on the screen rather than erratic, jumping numbers.
* **Intersection over Union (IoU):** A greedy matchmaker algorithm that reliably pairs live AI detections with existing tracked vehicles frame-by-frame.

## Tech Stack
* **Frontend / Android:** Kotlin, CameraX (for low-latency sensor access), Canvas API (for UI overlays).
* **Backend / Native:** C++, OpenCV, JNI (Java Native Interface). 

## How it Works (The Pipeline)
1. **Capture:** CameraX grabs a high-resolution frame and passes the raw memory buffer to C++ via JNI.
2. **Detect:** C++ normalizes the frame and feeds it through the YOLO model.
3. **Track & Correct:** The Kalman Filter guesses vehicle positions, and the AI detections correct those guesses.
4. **Calculate:** The system measures the ground distance the vehicle moved since the last frame and applies EMA to calculate a stable km/h speed.
5. **Render:** The coordinates are sent back to Kotlin, which maps the native sensor coordinates to the portrait screen and draws the visual overlay.
