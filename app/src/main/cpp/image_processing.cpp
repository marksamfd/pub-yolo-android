//
// Created by Esteban Uri on 17/01/2025.
//
#include <jni.h>
#include <android/log.h>
#include <cpu-features.h>

#define LOG_TAG "ImageProcessing"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Check for NEON support
//bool isNeonSupported() {
//    return (android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON) != 0;
//}
//
//// NEON optimized ARGB to RGB function
//void argb_to_rgb_neon(uint32_t* src, uint8_t* dest, int width, int height) {
//    // NEON optimized conversion code here
//    // This example doesn't contain actual NEON instructions but you can add NEON-specific operations.
//    // Assume each pixel is 32-bit ARGB, and the result is stored in RGB.
//
//    for (int i = 0; i < width * height; i++) {
//        uint32_t pixelValue = src[i];
//        uint8_t r = (pixelValue >> 16) & 0xFF;
//        uint8_t g = (pixelValue >> 8) & 0xFF;
//        uint8_t b = pixelValue & 0xFF;
//
//        dest[i * 3] = r;
//        dest[i * 3 + 1] = g;
//        dest[i * 3 + 2] = b;
//    }
//}



extern "C"
JNIEXPORT void JNICALL
Java_com_ultralytics_yolo_ImageProcessing_argb2yolo(
    JNIEnv *env,
    jobject thiz,
    jintArray srcArray,
    jobject destBuffer,
    jint width,
    jint height
) {

    // Get the source array
    jint* src = env->GetIntArrayElements(srcArray, 0);

    // Get the destination DirectByteBuffer (this will hold the float data)
    float* dest = (float*) env->GetDirectBufferAddress(destBuffer);

    if (dest == NULL) {
        // Handle error if the destination buffer is not a DirectByteBuffer
        return;
    }

    // Perform ARGB to RGB conversion and normalization to float (0.0f to 1.0f)
    for (int i = 0; i < width * height; ++i) {
        uint32_t pixel = src[i];  // ARGB format
        float r = ((pixel >> 16) & 0xFF) / 255.0f;
        float g = ((pixel >> 8) & 0xFF) / 255.0f;
        float b = (pixel & 0xFF) / 255.0f;

        // Store normalized float values for RGB channels
        int idx = i * 3;
        dest[idx] = r;     // Red
        dest[idx + 1] = g; // Green
        dest[idx + 2] = b; // Blue
    }

    // Release resources
    env->ReleaseIntArrayElements(srcArray, src, 0);
}