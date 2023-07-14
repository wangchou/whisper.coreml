#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoder256Wrapper.h"
#import "CoremlDecoder256.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inX;
MLMultiArray *inXa;
MLMultiArray *inQk_mask;

// output arrays
MLMultiArray *outX;
MLMultiArray *outQKs;
MLMultiArray *outMKV;
MLMultiArray *outCKV;

bool isPredicted = false;
bool isModelLoaded = false;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head) {
    CFTimeInterval startT = CACurrentMediaTime();
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (!isModelLoaded) {
        NSLog(@"loading %@", modelPathStr);
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    const void* model = CFBridgingRetain([[CoremlDecoder256 alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    int max_n_ctx = 256;
    // input arrays
    inX = getPixelBufferArray3(1, max_n_ctx, n_state);
    inXa = getPixelBufferArray3(1, 1500, n_state);
    inQk_mask = getPixelBufferArray2(max_n_ctx, max_n_ctx);

    outX = getPixelBufferArray3(1, max_n_ctx, n_state);
    outQKs = getPixelBufferArray4(n_layer, n_head, max_n_ctx, 1500);
    outMKV = getPixelBufferArray4(n_layer*2, 1, max_n_ctx, n_state);
    outCKV = getPixelBufferArray4(n_layer*2, 1, 1500, n_state);
    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* x, // (1, 256, n_state)
    float* xa, // (1, 1500, n_state)
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
    float32ToFloat16(x, (uint16*)inX.dataPointer, 1 * 256 * n_state);
    float32ToFloat16(xa, (uint16*)inXa.dataPointer, 1 * 1500 * n_state);
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 256 * 256);
    //NSLog(@"1 %.3f", CACurrentMediaTime() - startT);

    CoremlDecoder256Input* input = [[CoremlDecoder256Input alloc] initWithX:inX xa:inXa qk_mask:inQk_mask];

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

    output = (CoremlDecoder256Output*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    float16ToFloat32((uint16*)outX.dataPointer, out_x, outX.count);


    uint16* fromPtr = (uint16*)outQKs.dataPointer;
    float* toPtr = out_cross_qks;
    // ane fp16 output is aligned with 64 bytes or 32 element of fp16
    // 1500 is not multiple of 32 => ane appends 4 of zeors to 1504
    //showStrides(outQKs);
    int outQKsStride = [outQKs.strides[2] intValue];

    // This for loop is extramely slow, for small model
    // decoder256Test takes 57ms, this takes 11ms
    for(int i=0; i<n_layer * n_head * 256; i++) {
        float16ToFloat32(fromPtr, toPtr, 1500);
        fromPtr += outQKsStride;
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
}


void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inX.pixelBuffer);
    CFRelease(inXa.pixelBuffer);

    CFRelease(outX.pixelBuffer);
    CFRelease(outQKs.pixelBuffer);
    CFRelease(outMKV.pixelBuffer);
    CFRelease(outCKV.pixelBuffer);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif

