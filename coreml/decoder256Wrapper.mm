#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoder256Wrapper.h"
#import "CoremlDecoder256.h"
#include <stdlib.h>

void float32ToFloat16(const float* fp32, uint16* fp16, int count) {
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };

    //CFTimeInterval startT = CACurrentMediaTime();
    if (vImageConvert_PlanarFtoPlanar16F(&fp32Buffer, &fp16Buffer, 0) != kvImageNoError) {
        printf("float32toFloat16 error");
    }
    //NSLog(@"fp32tofp16 count=%d, time %.3f", count, CACurrentMediaTime() - startT);
}

void float16ToFloat32(const uint16* fp16, float* fp32, int count) {
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    if (vImageConvert_Planar16FtoPlanarF(&fp16Buffer, &fp32Buffer, 0) != kvImageNoError) {
        printf("float16toFloat32 error");
    }
}

// input arrays
MLMultiArray *inX;
MLMultiArray *inXa;
MLMultiArray *inQk_mask;

// output arrays
MLMultiArray *outX;
MLMultiArray *outQKs;
MLMultiArray *outMKV;
MLMultiArray *outCKV;

#if __cplusplus
extern "C" {
#endif

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
        NSLog(@"something wrong on creating PixelBuffer %d", cvRetval);
    }

    return pixelBuffer;
}

MLMultiArray* getPixelBufferArray2(int dim1, int dim2) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim2, dim1);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2)]];
}

MLMultiArray* getPixelBufferArray3(int dim1, int dim2, int dim3) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim3, dim1 * dim2);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3)]];
}

MLMultiArray* getPixelBufferArray4(int dim1, int dim2, int dim3, int dim4) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim4, dim1 * dim2 * dim3);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3), @(dim4)]];
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

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = 3;
    const void* model = CFBridgingRetain([[CoremlDecoder256 alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    int max_n_ctx = 256;
    // input arrays
    inX = getPixelBufferArray3(5, max_n_ctx, n_state);
    inXa = getPixelBufferArray3(5, 1500, n_state);
    inQk_mask = getPixelBufferArray2(max_n_ctx, max_n_ctx);

    // output arrays
    int f32_multiple = 2;
    outX = getPixelBufferArray3(5, max_n_ctx, n_state);
    // 1500 -> 1504 for being multiple of 32
    outQKs = getPixelBufferArray4(n_layer*5, n_head, max_n_ctx, 1504);
    outMKV = getPixelBufferArray4(n_layer*2, 5, max_n_ctx, n_state);
    outCKV = getPixelBufferArray4(n_layer*2, 1, 1500, n_state);
    return model;
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
}

bool isPredicted = false;

void predictWith(
    const void* model,
    float* x, // (bs, 256, n_state)
    float* xa, // (bs, 1500, n_state)
    float* qk_mask, // (256, 256)
    int n_layer,
    int n_state,
    int n_head,
    float* out_x,
    float* out_cross_qks,
    float* out_new_masked_kv_caches,
    float* out_new_cross_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToFloat16(x, (uint16*)inX.dataPointer, 5 * 256 * n_state);
    float32ToFloat16(xa, (uint16*)inXa.dataPointer, 5 * 1500 * n_state);
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 256 * 256);
    //NSLog(@"1 %.3f", CACurrentMediaTime() - startT);

    CoremlDecoder256Input* input = [[CoremlDecoder256Input alloc] initWithX:inX xa:inXa qk_mask:inQk_mask];

    // output arrays
    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_x":outX,
        @"out_cross_qks":outQKs,
        @"out_new_masked_kv_caches":outMKV,
        @"out_new_cross_kv_caches":outCKV
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoder256Output *output;
    //NSLog(@"2 %.3f", CACurrentMediaTime() - startT);

    output = (CoremlDecoder256Output*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];
    //output = (CoremlDecoder256Output*)[(__bridge id)model predictionFromFeatures:input error:&error];
    //NSLog(@"prediction %.3f", CACurrentMediaTime() - startT);
    if(error) {
        NSLog(@"%@", error);
    }

    //NSLog(@"3 %.3f", CACurrentMediaTime() - startT);
    float16ToFloat32((uint16*)outX.dataPointer, out_x, outX.count);


    //int dim1 = [outQKs.strides[0] intValue] / [outQKs.strides[1] intValue];
    //int dim2 = [outQKs.strides[1] intValue] / [outQKs.strides[2] intValue];
    //int dim3 = [outQKs.strides[2] intValue] / [outQKs.strides[3] intValue];
    //NSLog(@"dims = %d, %d, %d", dim1, dim2, dim3);

    // ane fp16 output is aligned with 64 bytes or 32 element of fp16
    // 1500 is not multiple of 32 => ane appends 4 of zeors to 1504
    uint16* fromPtr = (uint16*)outQKs.dataPointer;
    float* toPtr = out_cross_qks;
    for(int i=0; i<20 * 6 * 256; i++) {
        float16ToFloat32(fromPtr, toPtr, 1500);
        fromPtr += 1504;
        toPtr += 1500;
    }

    float16ToFloat32((uint16*)outMKV.dataPointer, out_new_masked_kv_caches, outMKV.count);
    float16ToFloat32((uint16*)outCKV.dataPointer, out_new_cross_kv_caches, outCKV.count);
    if (!isPredicted) {
        unlock(outX);
        unlock(outQKs);
        unlock(outMKV);
        unlock(outCKV);
        isPredicted = true;
    }
    //NSLog(@"4 %.3f", CACurrentMediaTime() - startT);
}


void closeModel(const void* model) {
   CFRelease(model);
   CFRelease(inX.pixelBuffer);
   CFRelease(inXa.pixelBuffer);

   CFRelease(outX.pixelBuffer);
   CFRelease(outQKs.pixelBuffer);
   CFRelease(outMKV.pixelBuffer);
   CFRelease(outCKV.pixelBuffer);
}

#if __cplusplus
} //Extern C
#endif
