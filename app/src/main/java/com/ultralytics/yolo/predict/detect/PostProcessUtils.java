package com.ultralytics.yolo.predict.detect;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Collections;

public class PostProcessUtils {

    // Sort objects in descending order based on confidence
    private static void qsortDescentInplace(ArrayList<DetectedObject> objects, int left, int right) {
        int i = left, j = right;
        float pivot = objects.get((left + right) / 2).confidence;

        while (i <= j) {
            while (objects.get(i).confidence > pivot) i++;
            while (objects.get(j).confidence < pivot) j--;
            if (i <= j) {
                // Swap
                Collections.swap(objects, i, j);
                i++;
                j--;
            }
        }

        if (left < j) qsortDescentInplace(objects, left, j);
        if (i < right) qsortDescentInplace(objects, i, right);
    }

    private static void qsortDescentInplace(ArrayList<DetectedObject> objects) {
        if (!objects.isEmpty()) {
            qsortDescentInplace(objects, 0, objects.size() - 1);
        }
    }

    // Calculate intersection area of two RectF objects
    private static float intersectionArea(RectF a, RectF b) {
        if (!RectF.intersects(a, b)) return 0;

        float left = Math.max(a.left, b.left);
        float right = Math.min(a.right, b.right);
        float top = Math.max(a.top, b.top);
        float bottom = Math.min(a.bottom, b.bottom);

        return Math.max(0, right - left) * Math.max(0, bottom - top);
    }

    // Non-Maximum Suppression (NMS)
    private static void nmsSortedBboxes(ArrayList<DetectedObject> objects, ArrayList<Integer> picked, float nmsThreshold) {
        picked.clear();

        int n = objects.size();
        ArrayList<Float> areas = new ArrayList<>(n);

        for (DetectedObject obj : objects) {
            areas.add(obj.boundingBox.width() * obj.boundingBox.height());
        }

        for (int i = 0; i < n; i++) {
            DetectedObject a = objects.get(i);
            boolean keep = true;

            for (int j : picked) {
                DetectedObject b = objects.get(j);

                // Intersection over Union (IoU)
                float interArea = intersectionArea(a.boundingBox, b.boundingBox);
                float unionArea = areas.get(i) + areas.get(j) - interArea;
                if (interArea / unionArea > nmsThreshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) picked.add(i);
        }
    }

    // Postprocess function
    public static ArrayList<DetectedObject> postprocess(
            float[][] recognitions,
            int w, int h,
            float confidenceThreshold,
            float iouThreshold,
            int numItemsThreshold,
            int numClasses,
            ArrayList<String> labels
    ) {

        ArrayList<DetectedObject> proposals = new ArrayList<>();
        ArrayList<DetectedObject> objects = new ArrayList<>();

        // Process recognitions
        for (int i = 0; i < w; i++) {
            float maxScore = -Float.MAX_VALUE;
            int classIndex = -1;

            for (int c = 0; c < numClasses; c++) {
                if (recognitions[c + 4][i] > maxScore) {
                    maxScore = recognitions[c + 4][i];
                    classIndex = c;
                }
            }

            if (maxScore > confidenceThreshold) {
                float dx = recognitions[0][i];
                float dy = recognitions[1][i];
                float dw = recognitions[2][i];
                float dh = recognitions[3][i];

                RectF rect = new RectF(
                        dx - dw / 2,
                        dy - dh / 2,
                        dx + dw / 2,
                        dy + dh / 2
                );
                String label = (labels != null && classIndex < labels.size()) ? labels.get(classIndex) : "Unknown";
                DetectedObject obj = new DetectedObject(maxScore, rect, classIndex, label);

                proposals.add(obj);
            }
        }

        // Sort proposals by confidence
        qsortDescentInplace(proposals);

        // Apply NMS
        ArrayList<Integer> picked = new ArrayList<>();
        nmsSortedBboxes(proposals, picked, iouThreshold);

        int count = Math.min(picked.size(), numItemsThreshold);
        for (int i = 0; i < count; i++) {
            objects.add(proposals.get(picked.get(i)));
        }

        // Convert to result format
        //float[][] result = new ArrayList<>();
        ArrayList<DetectedObject> result = new ArrayList<>();
        for (DetectedObject obj : objects) {

            float left = Math.max(0, obj.boundingBox.left);
            float top = Math.max(0, obj.boundingBox.top);
            float right = Math.min(1, obj.boundingBox.right);
            float bottom = Math.min(1, obj.boundingBox.bottom);

            RectF boundingBox = new RectF(
                left,
                top,
                right,
                bottom
            );

            DetectedObject det = new DetectedObject(
                obj.confidence,
                boundingBox,
                obj.index, obj.label
            );
            result.add(det);
            //result.add(new float[]{x, y, width, height, obj.confidence, obj.index});
        }

        return result;
    }
}
