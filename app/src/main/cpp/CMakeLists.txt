cmake_minimum_required(VERSION 3.4.1)

project("image_processing")

add_library(
        image_processing
        SHARED
        image_processing.cpp
)

# Specify include directories
include_directories(${ANDROID_NDK}/sources/android/cpufeatures)

# Search for prebuilt libraries
find_library( # Defines the name of the path variable.
        log-lib
        log
)

# Link the native library with log
target_link_libraries( # Specifies the target library.
        image_processing
        ${log-lib}
)
