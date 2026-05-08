#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <map>
#include <vector>
#include <set>


struct TrackedVehicle {
    int   id;
    int   classId;              // IDs: 0=person, 2=car, 5=bus, 7=truck

    // Variables
    float previousGroundX       = 0.f;
    float previousGroundY       = 0.f;
    long  previousTimestampNs   = 0;
    float currentSpeedKmh       = 0.f;
    float smoothedSpeedKmh      = 0.f;
    int   framesSinceLastSeen   = 0;
    cv::Rect lastBoundingBox;
    cv::Mat  templateImage;
    cv::KalmanFilter kf;
    bool kfInitialized          = false;
};

struct GroundPoint { float x; float y; };

class VehicleTracker {
public:
    VehicleTracker();

    void initModel(const std::string& onnxPath);

    std::vector<TrackedVehicle> processFrame(const cv::Mat& rgbaFrame,
                                             float pitchRad,
                                             long  timestampNs);
    void reset();

private:
    cv::dnn::Net net;
    std::map<int, TrackedVehicle> vehicleMap;
    int nextId      = 0;
    int frameCounter = 0;

    // CONSTANTS
    static constexpr int   DETECTION_INTERVAL   = 3;
    static constexpr int   MAX_FRAMES_MISSING   = 45;
    static constexpr float CONFIDENCE_THRESHOLD = 0.65f;
    static constexpr float IOU_MATCH_THRESHOLD  = 0.25f;
    static constexpr float TEMPLATE_MATCH_HIGH  = 0.75f;
    static constexpr float TEMPLATE_MATCH_LOW   = 0.50f;

    static constexpr float CAMERA_HEIGHT_METERS = 1.5f;
    static constexpr float CAMERA_V_FOV_DEG     = 60.0f;

    static constexpr float MAX_SPEED_KMH        = 250.f;
    static constexpr float MAX_PERSON_SPEED_KMH = 30.f;
    static constexpr float MIN_SPEED_DISPLAY    = 3.f;

    // Helper Functions
    cv::Rect sanitizeRect(cv::Rect r, int w, int h) const;
    float    computeIoU(const cv::Rect& a, const cv::Rect& b) const;
    float    calculateGroundDistance(float yPixel, int frameH, float pitchRad) const;
    void initKalman(TrackedVehicle& v, cv::Rect box);
    cv::Rect kalmanPredict(TrackedVehicle& v);
    void     kalmanCorrect(TrackedVehicle& v, cv::Rect measuredBox);
    void updateSpeed(TrackedVehicle& v, int frameW, int frameH, float pitchRad, long tsNs);
    GroundPoint calculateGroundPoint(float xPixel, float yPixel, int frameW, int frameH, float pitchRad) const;
};
