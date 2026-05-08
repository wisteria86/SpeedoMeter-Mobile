#include <jni.h>
#include <opencv2/core.hpp>
#include "VehicleTracker.h"
#include <android/log.h>

static VehicleTracker tracker;

extern "C" JNIEXPORT void JNICALL
Java_com_speedometer_MainActivity_initTracker(JNIEnv* env, jobject, jstring onnxPath) {
    const char* c_onnxPath = env->GetStringUTFChars(onnxPath, nullptr);
    tracker.initModel(c_onnxPath);
    env->ReleaseStringUTFChars(onnxPath, c_onnxPath);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_speedometer_MainActivity_processNativeFrame(
        JNIEnv* env, jobject, jobject rgbaBuffer, jint w, jint h, jint rowStride, jfloat pitch, jlong ts) {

    jfloatArray emptyArray = env->NewFloatArray(0);

    if (rgbaBuffer == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "speedometer", "Camera Buffer is Null!");
        return emptyArray;
    }

    void* addr = env->GetDirectBufferAddress(rgbaBuffer);

    cv::Mat rgbaFrame(h, w, CV_8UC4, addr, rowStride);

    std::vector<TrackedVehicle> vehicles;
    try {
        vehicles = tracker.processFrame(rgbaFrame, pitch, ts);
    } catch (const cv::Exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "speedometer", "OpenCV Crash: %s", e.what());
        return emptyArray;
    }

    if (vehicles.empty()) {
        return emptyArray;
    }

    int step = 6;
    jfloatArray out = env->NewFloatArray(vehicles.size() * step);
    std::vector<float> buffer(vehicles.size() * step);

    for (int i = 0; i < vehicles.size(); i++) {
        buffer[i*step + 0] = (float)vehicles[i].id;
        buffer[i*step + 1] = (float)vehicles[i].lastBoundingBox.x;
        buffer[i*step + 2] = (float)vehicles[i].lastBoundingBox.y;
        buffer[i*step + 3] = (float)vehicles[i].lastBoundingBox.width;
        buffer[i*step + 4] = (float)vehicles[i].lastBoundingBox.height;
        buffer[i*step + 5] = vehicles[i].currentSpeedKmh;
    }

    env->SetFloatArrayRegion(out, 0, (jsize)buffer.size(), buffer.data());

    return out;
}