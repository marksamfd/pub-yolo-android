# YoloEye Android App

An Android application that performs real-time door and obstacle detection using TensorFlow Lite and computer vision. Built upon TensorFlow's object detection example and Ultralytics' YOLOv11 implementation.

[![YoloEye Demo](https://img.youtube.com/vi/bg5JN4ZEdx4/1.jpg)](https://youtu.be/bg5JN4ZEdx4 "YoloEye Demo")
## Features

- Real-time object detection using multiple models:
  - YOLOv11
- Hardware acceleration support (CPU, GPU, NNAPI)
- Audio feedback for detected objects
- Configurable detection parameters
- Optimized for accessibility and navigation assistance

## Requirements

- Android Studio Meerkat (2024.3.1) or newer
- Android device with SDK 24+ (Android 7.0+)
- Developer mode enabled on device

## Installation

1. Clone the repository
2. Open project in Android Studio
3. Sync Gradle files
4. Build and run on physical Android device

## Configuration

The app allows customization of:
- Detection threshold
- Number of threads
- Maximum results
- Hardware delegate (CPU/GPU/NNAPI)
- Detection model

## License

This project is dual-licensed under:
- Apache License 2.0 (TensorFlow components)
- GNU GPL (YOLO components)

## Credits

Based on:
- [TensorFlow Lite Object Detection Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)
- [Ultralytics Flutter demo app](https://github.com/ultralytics/yolo-flutter-app)

For detailed implementation explanation, see [this Medium post](https://medium.com/p/6b7514556185).