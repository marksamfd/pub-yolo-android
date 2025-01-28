package com.ultralytics.yolo.predict.detect;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;


import androidx.camera.core.ImageProxy;

import com.ultralytics.yolo.ImageProcessing;
import com.ultralytics.yolo.ImageUtils;
import com.ultralytics.yolo.models.LocalYoloModel;
import com.ultralytics.yolo.models.YoloModel;
import com.ultralytics.yolo.predict.PredictorException;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;



public class TfliteDetector extends Detector {


    private static final long FPS_INTERVAL_MS = 1000; // Update FPS every 1000 milliseconds (1 second)
    private static final int NUM_BYTES_PER_CHANNEL = 4;
    private final Handler handler = new Handler(Looper.getMainLooper());
    // private final Matrix transformationMatrix;
    private final Bitmap pendingBitmapFrame;
    private int numClasses;
    private int frameCount = 0;
    private double confidenceThreshold = 0.25f;
    private double iouThreshold = 0.45f;
    private int numItemsThreshold = 30;
    private Interpreter interpreter;
    private Object[] inputArray;
    private int outputShape2;
    private int outputShape3;
    private float[][] output;
    private long lastFpsTime = System.currentTimeMillis();
    private Map<Integer, Object> outputMap;
    private ObjectDetectionResultCallback objectDetectionResultCallback;
    private FloatResultCallback inferenceTimeCallback;
    private FloatResultCallback fpsRateCallback;

    private static final float Nanos2Millis = 1 / 1e6f;
    public class Stats {
        private float imageSetupTime;
        private float inferenceTime;
        private float postProcessTime;

    }
    public Stats stats;

    private ByteBuffer imgData;
    private int[] intValues;
    private byte[] bytes;

    private ByteBuffer outData;

    private ByteBuffer pixelBuffer;

    private ImageProcessing ip;


    public TfliteDetector(Context context) {
        super(context);

        pendingBitmapFrame = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);


        stats = new Stats();

        imgData = null;
        intValues = null;
        outData = null;
        pixelBuffer = null;

        ip = new ImageProcessing();

    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        if (yoloModel instanceof LocalYoloModel) {
            final LocalYoloModel localYoloModel = (LocalYoloModel) yoloModel;

            if (localYoloModel.modelPath == null || localYoloModel.modelPath.isEmpty() ||
                    localYoloModel.metadataPath == null || localYoloModel.metadataPath.isEmpty()) {
                throw new Exception();
            }

            final AssetManager assetManager = context.getAssets();
            loadLabels(assetManager, localYoloModel.metadataPath);
            numClasses = labels.size();
            try {
                MappedByteBuffer modelFile = loadModelFile(assetManager, localYoloModel.modelPath);
                initDelegate(modelFile, useGpu);
            } catch (Exception e) {
                throw new PredictorException("Error model");
            }
        }
    }

    public Bitmap preprocess(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        return resizedBitmap;
    }



    @Override
    public ArrayList<DetectedObject> predict(Bitmap bitmap) {
        try {

            long startTime = System.nanoTime();
            // setInput(bitmap);
            setInputOptim(bitmap);
            stats.imageSetupTime = (System.nanoTime() - startTime) * Nanos2Millis;

            return runInference();
        } catch (Exception e) {
            return new ArrayList<>(); //float[0][];
        }
    }

    @Override
    public void setConfidenceThreshold(float confidence) {
        this.confidenceThreshold = confidence;
    }

    @Override
    public void setIouThreshold(float iou) {
        this.iouThreshold = iou;
    }

    @Override
    public void setNumItemsThreshold(int numItems) {
        this.numItemsThreshold = numItems;
    }

    @Override
    public void setObjectDetectionResultCallback(ObjectDetectionResultCallback callback) {
        objectDetectionResultCallback = callback;
    }

    @Override
    public void setInferenceTimeCallback(FloatResultCallback callback) {
        inferenceTimeCallback = callback;
    }

    @Override
    public void setFpsRateCallback(FloatResultCallback callback) {
        fpsRateCallback = callback;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {

        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    }

    private void initDelegate(MappedByteBuffer buffer, boolean useGpu) {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        try {
            // Check if GPU support is available
            CompatibilityList compatibilityList = new CompatibilityList();
            if (useGpu && compatibilityList.isDelegateSupportedOnThisDevice()) {
                GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions.setQuantizedModelsAllowed(true));
                interpreterOptions.addDelegate(gpuDelegate);
            } else {
                interpreterOptions.setNumThreads(4);
            }
            // Create the interpreter
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        } catch (Exception e) {
            interpreterOptions = new Interpreter.Options();
            interpreterOptions.setNumThreads(4);
            // Create the interpreter
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        }

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputShape2 = outputShape[1];
        outputShape3 = outputShape[2];
        output = new float[outputShape2][outputShape3];
    }

    private void setInput(Bitmap resizedbitmap) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * NUM_BYTES_PER_CHANNEL);
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        resizedbitmap.getPixels(intValues, 0, resizedbitmap.getWidth(), 0, 0, resizedbitmap.getWidth(), resizedbitmap.getHeight());

        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                float b = ((pixelValue & 0xFF)) / 255.0f;
                imgData.putFloat(r);
                imgData.putFloat(g);
                imgData.putFloat(b);
            }
        }
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        ByteBuffer outData = ByteBuffer.allocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
        outData.order(ByteOrder.nativeOrder());
        outData.rewind();
        outputMap.put(0, outData);
    }


    private void setInputOptim(Bitmap bitmap) {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (intValues == null) {
            intValues = new int[INPUT_SIZE * INPUT_SIZE];
            bytes = new byte[width * height * 3];

            int batchSize = 1;
            int RGB = 3;
            int numPixels = INPUT_SIZE * INPUT_SIZE;
            int bufferSize = batchSize * RGB * numPixels * NUM_BYTES_PER_CHANNEL;
            imgData = ByteBuffer.allocateDirect(bufferSize);
            imgData.order(ByteOrder.nativeOrder());

            outData = ByteBuffer.allocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
            outData.order(ByteOrder.nativeOrder());

        }
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        ip.argb2yolo(
                intValues,
                imgData,
                width,
                height
        );

        imgData.rewind();

        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        outData.rewind();
        outputMap.put(0, outData);

    }



    private ArrayList<DetectedObject> runInference() {
        if (interpreter != null) {

            long startTime = System.nanoTime();

            interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

            stats.inferenceTime = (System.nanoTime() - startTime) * Nanos2Millis;

            ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
            if (byteBuffer != null) {
                byteBuffer.rewind();

                for (int j = 0; j < outputShape2; ++j) {
                    for (int k = 0; k < outputShape3; ++k) {
                        output[j][k] = byteBuffer.getFloat();
                    }
                }


                startTime = System.nanoTime();

                ArrayList<DetectedObject> ret = PostProcessUtils.postprocess(
                        output,
                        outputShape3,
                        outputShape2,
                        (float) confidenceThreshold,
                        (float) iouThreshold,
                        numItemsThreshold,
                        numClasses,
                        labels
                );


                stats.postProcessTime = (System.nanoTime() - startTime) * Nanos2Millis;

                return ret;

            }
        }
        //return new float[0][];
        return new ArrayList<>();
    }




}
