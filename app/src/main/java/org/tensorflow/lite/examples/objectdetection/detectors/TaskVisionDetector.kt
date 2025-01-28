package org.tensorflow.lite.examples.objectdetection.detectors

import android.content.Context
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.Companion.MODEL_EFFICIENTDETV0
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.Companion.MODEL_EFFICIENTDETV1
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.Companion.MODEL_EFFICIENTDETV2
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.Companion.MODEL_MOBILENETV1
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.LinkedList

class TaskVisionDetector(
    var options: ObjectDetector.ObjectDetectorOptions,
    var currentModel: Int = 0,
    val context: Context,

    ): org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetector {

    private var objectDetector: ObjectDetector

    init {

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                else -> "mobilenetv1.tflite"
            }

        objectDetector = ObjectDetector.createFromFileAndOptions(context, modelName, options)

    }

    override fun detect(tensorImage: TensorImage, imageRotation: Int): DetectionResult {

        val tvDetections = objectDetector.detect(tensorImage)

        // Convert task view detections to common interface
        val detections = LinkedList<ObjectDetection>()
        for (tvDetection: Detection in tvDetections) {

            val cat = tvDetection.categories[0]

            val objDet = ObjectDetection(
                boundingBox = tvDetection.boundingBox,
                category = Category(
                    cat.label,
                    cat.score
                )
            )
            detections.add(objDet)
        }
        val results = DetectionResult(
            tensorImage.bitmap,
            detections
        )

        return results

    }
}