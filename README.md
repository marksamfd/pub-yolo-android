# TensorFlow Lite Object Detection Android Demo + YOLOv11

For detailed explanation see [this post](https://medium.com/p/6b7514556185).

![App example showing UI controls. Highlights a cat](pub/cat.jpeg)

![App example showing UI controls. Highlights a cat](pub/dog.gif)


### Overview

This application was adapted using code from:
- [TensorFlow Lite Object Detection Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android), and
- [Ultralytics Flutter demo app](https://github.com/ultralytics/yolo-flutter-app)

### Licences
Original TensorFlow Lite Object Detection Android Demo has [Apache License 2.0](LICENSE-Apache2.0.txt), while Ultralytics has [GNU GENERAL PUBLIC LICENSE](LICENSE).
So in case of using this code you must complain both licences.



### Application

This is a camera app that continuously detects the objects (bounding boxes and
classes) in the frames seen by your device's back camera, with the option to use
a quantized
[MobileNet SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2),
[EfficientDet Lite 0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1),
[EfficientDet Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1),
[EfficientDet Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1),
and [Ultralytics Yolo](https://docs.ultralytics.com/tasks/detect/#models)
model trained on the [COCO dataset](http://cocodataset.org/).


These instructions
walk you through building and running the demo on an Android device.

The model files are downloaded via Gradle scripts when you build and run the
app. You don't need to do any steps to download TFLite models into the project
explicitly.

This application should be run on a physical Android device.

![App example showing UI controls. Highlights a cat](pub/cat.gif)


## Build the demo using Android Studio

### Prerequisites

*   The **[Android Studio](https://developer.android.com/studio/index.html)**
    IDE. This sample has been tested on Android Studio Bumblebee.

*   A physical Android device with a minimum OS version of SDK 24 (Android 7.0 -
    Nougat) with developer mode enabled. The process of enabling developer mode
    may vary by device.

### Building

*   Open Android Studio. From the Welcome screen, select Open an existing
    Android Studio project.

*   From the Open File or Project window that appears, navigate to and select
    the tensorflow-lite/examples/object_detection/android directory. Click OK.

*   If it asks you to do a Gradle Sync, click OK.

*   With your Android device connected to your computer and developer mode
    enabled, click on the green Run arrow in Android Studio.


