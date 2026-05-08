cmake_minimum_required(VERSION 3.22.1)
# Note: I am naming the project speedometer to match your Android Studio path
project("speedometer")

# 1. Enforce C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 2. Define the exact path to the OpenCV native folder
set(OPENCV_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv/native")

# 3. Manually include the OpenCV header files so your code recognizes OpenCV syntax
include_directories("${OPENCV_NATIVE_DIR}/jni/include")

# 4. Manually import the pre-compiled OpenCV library (bypassing their broken CMake scripts)
add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES
        IMPORTED_LOCATION "${OPENCV_NATIVE_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so")

# 5. Compile your native code
add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        VehicleTracker.cpp)

# 6. Link your code with OpenCV, Android hardware, and logs
target_link_libraries(${CMAKE_PROJECT_NAME}
        opencv_java4
        android
        log)