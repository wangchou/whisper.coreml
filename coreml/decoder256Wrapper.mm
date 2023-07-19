#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoder256Wrapper.h"
#import "CoremlDecoder256.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inX;
MLMultiArray *inQk_mask;
MLMultiArray *inCk;
MLMultiArray *inCv;

// output arrays
MLMultiArray *outX;
MLMultiArray *outCHW; // cross_head_weights
MLMultiArray *outMKV;

bool isPredicted = false;
bool isModelLoaded = false;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_alignment_head) {
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
    inQk_mask = getPixelBufferArray2(max_n_ctx, max_n_ctx);
    inCk = getPixelBufferArray4(n_layer, n_head, 64, 1500);
    inCv = getPixelBufferArray4(n_layer, n_head, 1500, 64);

    outX = getPixelBufferArray3(1, max_n_ctx, n_state);
    outCHW = getPixelBufferArray3(n_alignment_head, max_n_ctx, 1500);
    outMKV = getPixelBufferArray4(n_layer*2, 1, max_n_ctx, n_state);
    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* x, // (1, 256, n_state)
    float* qk_mask, // (256, 256)
    float* cross_k_caches,
    float* cross_v_caches,
    bool isNewCKV,
    float* out_x,
    float* out_cross_head_weights,
    float* out_new_masked_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToMa(x, inX);
    float32ToMa(qk_mask, inQk_mask);
    if (isNewCKV) {
        float32ToMa(cross_k_caches, inCk);
        float32ToMa(cross_v_caches, inCv);
    }
    //NSLog(@"1 %.3f", CACurrentMediaTime() - startT);

    CoremlDecoder256Input* input = [[CoremlDecoder256Input alloc] initWithX:inX qk_mask:inQk_mask cross_k_caches:inCk cross_v_caches:inCv];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_x":outX,
        @"out_cross_head_weights":outCHW,
        @"out_new_masked_kv_caches":outMKV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoder256Output *output;

    output = (CoremlDecoder256Output*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    maToFloat32(outX, out_x);

    maToFloat32(outCHW, out_cross_head_weights);

    maToFloat32(outMKV, out_new_masked_kv_caches);

    if (!isPredicted) {
        unlock(outX);
        unlock(outCHW);
        unlock(outMKV);
        isPredicted = true;
    }
}


void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inX.pixelBuffer);
    CFRelease(inCk.pixelBuffer);
    CFRelease(inCv.pixelBuffer);

    CFRelease(outX.pixelBuffer);
    CFRelease(outCHW.pixelBuffer);
    CFRelease(outMKV.pixelBuffer);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif

