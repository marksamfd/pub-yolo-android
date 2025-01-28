package com.ultralytics.yolo;
import java.nio.ByteBuffer;

public class ImageProcessing {

    static {
        System.loadLibrary("image_processing");
    }

    public native void argb2yolo(
            int[] src,
            ByteBuffer dest,
            int width,
            int height
    );


}
