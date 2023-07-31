#import "coremlUtility.h"
#include <CoreML/CoreML.h>

void float32ToFloat16(const float* fp32, const uint16* fp16, int count) {
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };

    //CFTimeInterval startT = CACurrentMediaTime();
    if (vImageConvert_PlanarFtoPlanar16F(&fp32Buffer, &fp16Buffer, 0) != kvImageNoError) {
        printf("float32toFloat16 error");
    }
    //NSLog(@"fp32tofp16 count=%d, time %.3f", count, CACurrentMediaTime() - startT);
}

void float16ToFloat32(const uint16* fp16, const float* fp32, int count) {
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    if (vImageConvert_Planar16FtoPlanarF(&fp16Buffer, &fp32Buffer, 0) != kvImageNoError) {
        printf("float16toFloat32 error");
    }
}

void float32ToMa(const float* fp32, MLMultiArray* ma) {
    int n_dim = ma.shape.count;
    int maStride = [ma.strides[n_dim-2] intValue];
    int fp32Stride = [ma.shape[n_dim-1] intValue];
    bool isAligned = maStride == fp32Stride;

    uint16* fp16 = (uint16*) ma.dataPointer;
    if (isAligned) {
        float32ToFloat16(fp32, fp16, ma.count);
        return;
    }

    int sliceCount = ma.count / fp32Stride;
    float* _fp32 = (float*) fp32;
    for(int i=0; i<sliceCount; i++) {
        float32ToFloat16(_fp32, fp16, fp32Stride);
        _fp32 += fp32Stride;
        fp16 += maStride;
    }
}

void maToFloat32(MLMultiArray* ma, const float* fp32) {
    int n_dim = ma.shape.count;
    int maStride = [ma.strides[n_dim-2] intValue];
    int fp32Stride = [ma.shape[n_dim-1] intValue];
    bool isAligned = maStride == fp32Stride;

    uint16* fp16 = (uint16*) ma.dataPointer;
    if (isAligned) {
        float16ToFloat32(fp16, fp32, ma.count);
        return;
    }

    int sliceCount = ma.count / fp32Stride;
    int fp16Stride = maStride;
    float* _fp32 = (float*) fp32;
    for(int i=0; i<sliceCount; i++) {
        float16ToFloat32(fp16, _fp32, fp32Stride);
        _fp32 += fp32Stride;
        fp16 += maStride;
    }
}

void unlock(MLMultiArray* ma) {
    CVReturn cvRetval = 0;
    cvRetval = CVPixelBufferUnlockBaseAddress(ma.pixelBuffer, 0);

    if (cvRetval != kCVReturnSuccess) {
        NSLog(@"something wrong on unlocking PixelBuffer %d", cvRetval);
    }
}

void showStrides(MLMultiArray* ma) {
    NSLog(@" ");
    NSLog(@"count %ld %f", ma.count, ma.count / [ma.strides[0] floatValue]);
    for(int i=0; i<ma.strides.count; i++) {
        NSLog(@"stride %d %@", i, ma.strides[i]);
    }
    for(int i=0; i<ma.shape.count; i++) {
        NSLog(@"shape %d %@", i, ma.shape[i]);
    }
}

CVPixelBufferRef getPixelBuffer(int dim1, int dim2) {
    CVPixelBufferRef pixelBuffer = NULL;
    CVReturn cvRetval = 0;
    NSDictionary* poptions = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
    cvRetval = CVPixelBufferCreate(
            kCFAllocatorDefault,
            dim1, dim2,
            kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)poptions,
            &pixelBuffer);

    if (cvRetval != kCVReturnSuccess) {
        NSLog(@"something wrong on creating PixelBuffer %d, dim1=%d, dim2=%d", cvRetval, dim1, dim2);
    }

    return pixelBuffer;
}

MLMultiArray* getPixelBufferArray2(int dim1, int dim2) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim2, dim1);
    MLMultiArray* ma = [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2)]];

    // unlock once to avoid it locked by dataPointer later
    void * ptr = ma.dataPointer;
    unlock(ma);
    return ma;
}

MLMultiArray* getPixelBufferArray3(int dim1, int dim2, int dim3) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim3, dim1 * dim2);
    MLMultiArray* ma = [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3)]];
    void * ptr = ma.dataPointer;
    unlock(ma);
    return ma;
}

MLMultiArray* getPixelBufferArray4(int dim1, int dim2, int dim3, int dim4) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim4, dim1 * dim2 * dim3);
    MLMultiArray* ma = [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3), @(dim4)]];
    void * ptr = ma.dataPointer;
    unlock(ma);
    return ma;
}

MLMultiArray* getArray1(void* dataPtr, int dim1, MLMultiArrayDataType dataType) {
    return [[MLMultiArray alloc]
        initWithDataPointer: dataPtr
        shape: @[@(dim1)]
        dataType: dataType
        strides: @[@1]
        deallocator: nil
        error: nil
    ];
}
