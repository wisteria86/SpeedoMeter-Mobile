#include "VehicleTracker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <android/log.h>
#include <cmath>
#include <algorithm>

#define LOG_TAG "VehicleTracker"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;


VehicleTracker::VehicleTracker() { reset(); }

void VehicleTracker::initModel(const string& onnxPath) {
    // Using MultiThread Here for Efficiency
    setNumThreads(4);

    // Loading Model (my model weighst are in .onnx format)
    net = dnn::readNetFromONNX(onnxPath);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    LOGI("Model loaded. Empty=%d", net.empty());
}

// Resets Main Variables and Array
void VehicleTracker::reset() {
    vehicleMap.clear();
    nextId       = 0;
    frameCounter = 0;
}

// This makes box stay strictly within the boundaries of an image by
// clipping off any parts that go out of bounds.
Rect VehicleTracker::sanitizeRect(Rect r, int w, int h) const {
    return r & Rect(0, 0, w, h);
}

// Intersection over Union (IoU)
// Calculates Overlap between Boxes
float VehicleTracker::computeIoU(const Rect& a, const Rect& b) const {
    // intersection
    Rect inter = a & b;
    // zero overlap check
    if (inter.area() <= 0) return 0.f;
    // IoU Ratio Formula (Area of Overlap / Area of Union)
    return (float)inter.area() / (float)(a.area() + b.area() - inter.area());
}


// Converts 2D pixel into 3D Ground Distance (in Meters)
float VehicleTracker::calculateGroundDistance(float yPixel, int frameH, float pitchRad) const {
    // Converting the vertical Field of View from degrees to radians.
    float vFovRad   = CAMERA_V_FOV_DEG * (float)M_PI / 180.f;
    // Vertical Center of Image
    float cy        = frameH / 2.f;
    // Calculating Focal Length in pixels. fy is derived from the half-angle of the FOV and the half-height of the image
    float fy        = cy / tan(vFovRad / 2.f);
    // Determines the angle of the specific pixel relative to the camera's optical center
    float pixelAngle = atan2f(yPixel - cy, fy);
    // Adding Pitch to the Pixel Angle Cause Camera isn't always pointing Perfectly.
    float totalAngle = pitchRad + pixelAngle;
    if (totalAngle <= 0.01f) return -1.f;
    // Triangulating The Distance
    return CAMERA_HEIGHT_METERS / tan(totalAngle);
}

// Converts 2D pixel into 3D Ground Distance (in Meters)
GroundPoint VehicleTracker::calculateGroundPoint(float xPixel, float yPixel, int frameW, int frameH, float pitchRad) const {
    // Horizontal and Vertical FOV (In Radians)
    float vFovRad = CAMERA_V_FOV_DEG * (float)M_PI / 180.f;
    float hFovRad = vFovRad * ((float)frameW / frameH);

    // Center of Image
    float cx = frameW / 2.f;
    float cy = frameH / 2.f;

    float fy = cy / tan(vFovRad / 2.f);
    float fx = cx / tan(hFovRad / 2.f);

    float angleY = atan2f(yPixel - cy, fy) + pitchRad;
    float angleX = atan2f(xPixel - cx, fx);

    if (angleY <= 0.01f) return {0.f, -1.f}; // Looking at horizon

    // Depth (Distance from camera along the ground)
    float groundY = CAMERA_HEIGHT_METERS / tan(angleY);
    // Lateral (Side-to-side distance on the ground)
    float groundX = groundY * tan(angleX);

    return {groundX, groundY};
}


// Kalman basically determines the position of object in next Frame, even if there is noise present.
void VehicleTracker::initKalman(TrackedVehicle& v, Rect box) {
    // 6 states: [cx, cy, w, h, vx, vy], 4 measurements: [cx, cy, w, h]
    v.kf.init(6, 4, 0, CV_32F);

    v.kf.transitionMatrix = (Mat_<float>(6, 6) <<
                                               1, 0, 0, 0, 1, 0,  // cx = cx + vx
            0, 1, 0, 0, 0, 1,  // cy = cy + vy
            0, 0, 1, 0, 0, 0,  // w constant (smooths over time)
            0, 0, 0, 1, 0, 0,  // h constant (smooths over time)
            0, 0, 0, 0, 1, 0,  // vx constant
            0, 0, 0, 0, 0, 1); // vy constant

    // Tell filter to map 4 measurements to the first 4 states directly
    setIdentity(v.kf.measurementMatrix);
    setIdentity(v.kf.processNoiseCov, Scalar::all(1e-2f));
    setIdentity(v.kf.measurementNoiseCov, Scalar::all(0.5f));
    setIdentity(v.kf.errorCovPost, Scalar::all(1.0f));

    // Initialize with Pixels.
    v.kf.statePost.at<float>(0) = (float)(box.x + box.width / 2);
    v.kf.statePost.at<float>(1) = (float)(box.y + box.height / 2);
    v.kf.statePost.at<float>(2) = (float)box.width;
    v.kf.statePost.at<float>(3) = (float)box.height;
    v.kf.statePost.at<float>(4) = 0.f;
    v.kf.statePost.at<float>(5) = 0.f;

    v.kfInitialized = true;
}

// Predicts Next Position and Velocity of Object.
Rect VehicleTracker::kalmanPredict(TrackedVehicle& v) {
    Mat pred = v.kf.predict();

    int cx = (int)pred.at<float>(0);
    int cy = (int)pred.at<float>(1);
    int w  = (int)pred.at<float>(2);
    int h  = (int)pred.at<float>(3);

    // Safety fallback so boxes don't collapse on bad predictions
    w = max(10, w);
    h = max(10, h);

    return Rect(cx - w / 2, cy - h / 2, w, h);
}


// Uses actual visual data to ground Kalman's guess in reality.
void VehicleTracker::kalmanCorrect(TrackedVehicle& v, Rect measuredBox) {
    // Taking Measurements
    Mat meas(4, 1, CV_32F);
    meas.at<float>(0) = (float)(measuredBox.x + measuredBox.width  / 2);
    meas.at<float>(1) = (float)(measuredBox.y + measuredBox.height / 2);
    meas.at<float>(2) = (float)measuredBox.width;
    meas.at<float>(3) = (float)measuredBox.height;

    // Error check
    Mat corrected = v.kf.correct(meas);

    // Updating
    int cx = (int)corrected.at<float>(0);
    int cy = (int)corrected.at<float>(1);
    int w  = (int)corrected.at<float>(2);
    int h  = (int)corrected.at<float>(3);

    w = max(10, w);
    h = max(10, h);

    v.lastBoundingBox = Rect(cx - w / 2, cy - h / 2, w, h);
}

// Finds Speed of Object
void VehicleTracker::updateSpeed(TrackedVehicle& v, int frameW, int frameH, float pitchRad, long tsNs) {
    float deltaT = (tsNs - v.previousTimestampNs) / 1e9f;

    if (deltaT < 0.03f || deltaT > 2.0f) return;

    // Get current ground position
    GroundPoint currGP = calculateGroundPoint(v.lastBoundingBox.x + v.lastBoundingBox.width/2.f,
                                              v.lastBoundingBox.y + v.lastBoundingBox.height,
                                              frameW, frameH, pitchRad);

    if (currGP.y < 0) return;

    // Calculates Euclidean distance (Horizontal + Vertical)
    float dx = currGP.x - v.previousGroundX;
    float dy = currGP.y - v.previousGroundY;
    float displacement = sqrtf(dx*dx + dy*dy);

    // Checking displacement vs noise
    if (displacement < 0.15f) { // sensitive threshold
        v.smoothedSpeedKmh *= 0.8f;
        if (v.smoothedSpeedKmh < 1.0f) v.smoothedSpeedKmh = 0.f;
    } else {
        float rawSpeed = (displacement / deltaT) * 3.6f;
        float cap = (v.classId == 0) ? 15.f : 160.f;
        rawSpeed = std::min(rawSpeed, cap);

        // Exponential Moving Average
        float alpha = 0.15f;
        v.smoothedSpeedKmh = alpha * rawSpeed + (1.f - alpha) * v.smoothedSpeedKmh;
    }

    v.currentSpeedKmh = v.smoothedSpeedKmh;
    v.previousGroundX = currGP.x;
    v.previousGroundY = currGP.y;
    v.previousTimestampNs = tsNs;
}

vector<TrackedVehicle> VehicleTracker::processFrame(const Mat& rgbaFrame, float pitchRad, long timestampNs) {

    // Base Check for Neural Model.
    if (net.empty()) {
        LOGE("Net is empty – call initModel first!");
        return {};
    }

    const int sensorW = rgbaFrame.cols;
    const int sensorH = rgbaFrame.rows;

    // Grayscale version of the frame.
    // We use this later for "template matching" tracking step.
    Mat grayForTracking;
    cvtColor(rgbaFrame, grayForTracking, COLOR_RGBA2GRAY);

    // Keep track of frames so we know when to run the heavy neural network.
    frameCounter++;
    bool isDetectionFrame = (frameCounter % DETECTION_INTERVAL == 0);


    // Asking Kalman to Predict our current tracked Objects
    for (auto& [id, v] : vehicleMap) {
        if (v.kfInitialized) {
            Rect predicted = kalmanPredict(v);
            // Makes sure the predicted box doesn't accidentally wander off the edge of the screen.
            v.lastBoundingBox = sanitizeRect(predicted, sensorW, sensorH);
        }
    }

    set<int> matchedIds; // Place a Tag on Each Tracked Object


    if (isDetectionFrame) {
        // Running the DNN is computationally expensive, so we only do it every few frames.
        // The network expects the image format: BGR, rotated, and perfectly square.
        Mat bgrFrame;
        cvtColor(rgbaFrame, bgrFrame, COLOR_RGBA2BGR);

        int maxDim = max(bgrFrame.cols, bgrFrame.rows);
        Mat squareFrame = Mat::zeros(maxDim, maxDim, CV_8UC3);
        bgrFrame.copyTo(squareFrame(Rect(0, 0, bgrFrame.cols, bgrFrame.rows)));

        // Convert the image into a "blob" (a multi-dimensional array) scaled down to 640x640.
        Mat blob = dnn::blobFromImage(
                squareFrame, 1.0/255.0, Size(640, 640),
                Scalar(0,0,0), true, false);
        net.setInput(blob);

        // Call Neural Network's Forward Pass.
        Mat output = net.forward();

        // Flatten the output tensor into a 2D matrix so it's easier to read.
        Mat detections(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        detections = detections.t();

        const int rotW = sensorH;
        const int rotH = sensorW;

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        float scale_factor = (float)maxDim / 640.0f;

        // Parsing the Network's Guesses
        for (int i = 0; i < detections.rows; ++i) {
            float* row_ptr = detections.ptr<float>(i);

            // The network outputs probabilities for 80 different object classes.
            // We want to find the specific class it's most confident about.
            Mat scores(1, 80, CV_32F, row_ptr + 4);
            Point classIdPoint;
            double max_class_score;
            minMaxLoc(scores, nullptr, &max_class_score, nullptr, &classIdPoint);

            // Checks If the network is reasonably sure about what it saw.
            if (max_class_score > CONFIDENCE_THRESHOLD) {
                int classId = classIdPoint.x;

                // ID: 0=person, 2=car, 5=bus, 7=truck.
                if (classId != 0 && classId != 2 && classId != 5 && classId != 7) continue;

                // Convert the bounding box coordinates from the 640x640 network space
                // back up to our original camera resolution space.
                float cx = row_ptr[0] * scale_factor;
                float cy = row_ptr[1] * scale_factor;
                float w  = row_ptr[2] * scale_factor;
                float h  = row_ptr[3] * scale_factor;

                int left = int(cx - 0.5f * w);
                int top  = int(cy - 0.5f * h);

                classIds.push_back(classId);
                confidences.push_back((float)max_class_score);
                boxes.push_back(Rect(left, top, int(w), int(h)));
            }
        }


        // Non-Maximum Suppression And De-Rotation
        // The network often guesses the exact same car multiple times.
        // NMS picks the single best bounding box and throws away the overlapping duplicates.
        vector<int> nms_indices;
        dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.3f, nms_indices);

        vector<pair<Rect, int>> detectedBoxes;
        for (int idx : nms_indices) {
            Rect r = boxes[idx];
            int classId = classIds[idx];

            Rect sensorRect = sanitizeRect(r, sensorW, sensorH);

            // Filters out tiny noise boxes.
            if (sensorRect.area() < 10000) continue;

            detectedBoxes.push_back({sensorRect, classId});
        }

        // Matchmaker (Assigning Detections to Trackers)
        set<int> assignedTrackers;
        for (auto& [box, cls] : detectedBoxes) {
            float bestIoU  = IOU_MATCH_THRESHOLD;
            int   bestId   = -1;

            // Check this new detection against all the vehicles we are already tracking.
            // We use IoU (Intersection over Union) to see if the boxes overlap.
            for (auto& [id, v] : vehicleMap) {
                if (assignedTrackers.count(id)) continue;
                float iou = computeIoU(box, v.lastBoundingBox);
                if (iou > bestIoU) { bestIoU = iou; bestId = id; }
            }

            // We found a duplicate
            if (bestId != -1) {
                assignedTrackers.insert(bestId);
                TrackedVehicle& v = vehicleMap[bestId];

                // Uses the real visual detection to correct our Kalman physics guess.
                kalmanCorrect(v, box);
                v.lastBoundingBox    = sanitizeRect(v.lastBoundingBox, sensorW, sensorH);
                v.framesSinceLastSeen = 0; // Resetting the "missing" counter.
                v.classId            = cls;

                // Save a tiny grayscale picture of the car. We'll use this image patch
                // to track the car on frames where we don't run the heavy neural network.
                Rect safeBox = sanitizeRect(v.lastBoundingBox, sensorW, sensorH);
                if (safeBox.area() > 0)
                    v.templateImage = grayForTracking(safeBox).clone();

                matchedIds.insert(bestId);
            } else {
                // No Match. So it's not a duplicate.
                TrackedVehicle nv;
                nv.id                 = nextId++;
                nv.classId            = cls;
                nv.lastBoundingBox    = box;

                // Calculates its exact 3D position on the ground plane.
                GroundPoint gp = calculateGroundPoint(box.x + box.width / 2.f, box.y + box.height, sensorW, sensorH, pitchRad);
                nv.previousGroundX    = gp.x;
                nv.previousGroundY    = gp.y;

                nv.previousTimestampNs = timestampNs;
                nv.framesSinceLastSeen = 0;
                nv.templateImage      = grayForTracking(sanitizeRect(box, sensorW, sensorH)).clone();

                // Boot up a fresh physics tracker for it.
                initKalman(nv, box);
                vehicleMap[nv.id]     = nv;
                matchedIds.insert(nv.id);
                LOGI("New track id=%d cls=%d", nv.id, cls);
            }
        }

        // If any vehicle wasn't matched with a new detection, mark it as unseen.
        for (auto& [id, v] : vehicleMap) {
            if (matchedIds.find(id) == matchedIds.end()) {
                v.framesSinceLastSeen++;
            }
        }
    }

        // Template Matching
    else {
        // This is a "fast" frame. So, We skip the heavy neural network.
        // Instead, we use classical computer vision to find the objects :D.
        for (auto& [id, v] : vehicleMap) {
            if (v.templateImage.empty()) {
                v.framesSinceLastSeen++;
                continue;
            }

            // A "search window" around where the object was last seen.
            // We assume the object hasn't teleported, so it must be nearby.
            int pad = 40;
            Rect searchWindow = sanitizeRect(
                    Rect(v.lastBoundingBox.x - pad,
                         v.lastBoundingBox.y - pad,
                         v.lastBoundingBox.width  + pad * 2,
                         v.lastBoundingBox.height + pad * 2),
                    sensorW, sensorH);

            if (searchWindow.width  <= v.templateImage.cols ||
                searchWindow.height <= v.templateImage.rows) {
                v.framesSinceLastSeen++;
                continue;
            }

            Mat searchArea = grayForTracking(searchWindow);
            Mat matchResult;

            // Slide the saved image patch over the search window to find the best pixel match.
            matchTemplate(searchArea, v.templateImage, matchResult, TM_CCOEFF_NORMED);

            double maxVal; Point maxLoc;
            minMaxLoc(matchResult, nullptr, &maxVal, nullptr, &maxLoc);

            // High Confidence Match Check
            if (maxVal >= TEMPLATE_MATCH_HIGH) {
                Rect newBox(
                        searchWindow.x + maxLoc.x,
                        searchWindow.y + maxLoc.y,
                        v.lastBoundingBox.width,
                        v.lastBoundingBox.height);
                newBox = sanitizeRect(newBox, sensorW, sensorH);

                // Corrects the Kalman filter with this new visual position.
                kalmanCorrect(v, newBox);

                v.lastBoundingBox     = sanitizeRect(v.lastBoundingBox, sensorW, sensorH);
                v.framesSinceLastSeen = 0;
            } else if (maxVal >= TEMPLATE_MATCH_LOW) {
                // Weak match. It might be the object, but we aren't sure enough so we do nothing.
            } else {
                // No match at all.
                v.framesSinceLastSeen++;
            }
        }
    }

    // Speed Updates & Cleanup
    for (auto& [id, v] : vehicleMap) {
        if (v.framesSinceLastSeen == 0) {
            // calculate Speed
            updateSpeed(v, sensorW, sensorH, pitchRad, timestampNs);
        }
    }

    // if object has been missing for too long, we assume it has left the scene.
    for (auto it = vehicleMap.begin(); it != vehicleMap.end();) {
        it = (it->second.framesSinceLastSeen > MAX_FRAMES_MISSING)
             ? vehicleMap.erase(it) : ++it;
    }

    // Here, we Package up only the objects we are actively tracking right now to return to the UI.
    // We allow a grace peroid where we still return the object,
    vector<TrackedVehicle> results;
    results.reserve(vehicleMap.size());
    for (auto& [id, v] : vehicleMap) {
        if (v.framesSinceLastSeen <= 5) {
            results.push_back(v);
        }
    }

    return results;
}