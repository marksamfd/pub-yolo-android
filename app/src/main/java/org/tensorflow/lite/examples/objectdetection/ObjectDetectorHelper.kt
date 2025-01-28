/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.task.core.BaseOptions

import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector

import org.tensorflow.lite.examples.objectdetection.detectors.TaskVisionDetector
import org.tensorflow.lite.examples.objectdetection.detectors.YoloDetector
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions


class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }


    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {

        try {

            if (currentModel == MODEL_YOLO) {

                objectDetector = YoloDetector(

                    threshold,
                    0.3f,
                    numThreads,
                    maxResults,
                    currentDelegate,
                    currentModel,
                    context,

                )

            }
            else {

                // Create the base options for the detector using specifies max results and score threshold
                val optionsBuilder =
                    ObjectDetectorOptions.builder()
                        .setScoreThreshold(threshold)
                        .setMaxResults(maxResults)

                // Set general detection options, including number of used threads
                val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

                // Use the specified hardware for running the model. Default to CPU
                when (currentDelegate) {
                    DELEGATE_CPU -> {
                        // Default
                    }
                    DELEGATE_GPU -> {
//                        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
//                            baseOptionsBuilder.useGpu()
//                        } else {
//                            objectDetectorListener?.onError("GPU is not supported on this device")
//                        }
                        // for some reason CompatibilityList().isDelegateSupportedOnThisDevice
                        // returns False in my Motorola Edge 30 Ultra, but GPU works :/
                        baseOptionsBuilder.useGpu()
                    }
                    DELEGATE_NNAPI -> {
                        baseOptionsBuilder.useNnapi()
                    }
                }

                optionsBuilder.setBaseOptions(baseOptionsBuilder.build())
                val options = optionsBuilder.build()

                objectDetector = TaskVisionDetector(
                    options,
                    currentModel,
                    context,

                )

            }


        }
        catch (e : Exception) {

            objectDetectorListener?.onError(e.toString())

        }


    }


    fun detect(image: Bitmap, imageRotation: Int) {

        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/lite_support#imageprocessor_architecture

        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        val results = objectDetector?.detect(tensorImage, imageRotation)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (results != null) {
            objectDetectorListener?.onResults(
                results.detections,
                inferenceTime,
                results.image.height,
                results.image.width
            )
        }

    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: List<ObjectDetection>,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_YOLO = 4
    }
}
