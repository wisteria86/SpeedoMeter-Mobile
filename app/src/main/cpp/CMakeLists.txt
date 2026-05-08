cmake_minimum_required(VERSION 3.22.1)
project("speedometer")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OPENCV_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../opencv/native")

include_directories("${OPENCV_NATIVE_DIR}/jni/include")

add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES
        IMPORTED_LOCATION "${OPENCV_NATIVE_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so")

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        VehicleTracker.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        opencv_java4
        android
        log)
