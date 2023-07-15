#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoderWrapper.h"
#import "CoremlDecoder.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inX;
MLMultiArray *inQk_mask;
MLMultiArray *inMkv;
MLMultiArray *inCkv;

// output arrays
MLMultiArray *outX;
MLMultiArray *outMKV;

bool isPredicted = false;
bool isModelLoaded = false;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab) {
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
    const void* model = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    // input arrays
    inX = getPixelBufferArray3(5, 1, n_state);
    inQk_mask = getPixelBufferArray2(1, 449);
    inMkv = getPixelBufferArray4(n_layer*2, 5, 448, n_state);
    inCkv = getPixelBufferArray4(n_layer*2, 1, 1500, n_state);

    // output arrays
    outX = getPixelBufferArray3(5, 1, n_vocab);
    outMKV = getPixelBufferArray4(n_layer*2, 5, 1, n_state);

    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_kv_caches, // (n_layer * 2, 1, 1500, n_state)
    int n_layer,
    int n_state,
    int n_head,
    int n_vocab,
    bool isNewCKV,
    float* out_x,
    float* out_new_masked_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToFloat16(x, (uint16*)inX.dataPointer, 5 * n_state);
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 449);
    float32ToFloat16(masked_kv_caches, (uint16*)inMkv.dataPointer, n_layer * 2 * 5 * 448 * n_state);

    if (isNewCKV) {
        float32ToFloat16(cross_kv_caches, (uint16*)inCkv.dataPointer, n_layer * 2 * 1 * 1500 * n_state);
    }

    CoremlDecoderInput* input = [[CoremlDecoderInput alloc] initWithX:inX qk_mask:inQk_mask masked_kv_caches:inMkv cross_kv_caches:inCkv];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_x":outX,
        @"out_new_masked_kv_caches":outMKV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoderOutput *output;

    output = (CoremlDecoderOutput*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    // ane fp16 output is aligned with 64 bytes or 32 element of fp16
    // 51865 is not multiple of 32 => ane appends zeors to 51872
    //showStrides(outX);
    uint16* fromPtr = (uint16*)outX.dataPointer;
    float* toPtr = out_x;
    int outXStride = [outX.strides[0] intValue];
    for(int bs=0; bs<5; bs++) {
        float16ToFloat32(fromPtr, toPtr, n_vocab);
        fromPtr += outXStride;
        toPtr += n_vocab;
    }

    float16ToFloat32((uint16*)outMKV.dataPointer, out_new_masked_kv_caches, outMKV.count);
    if (!isPredicted) {
        unlock(outX);
        unlock(outMKV);
        isPredicted = true;
    }
}

void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inX.pixelBuffer);
    CFRelease(inQk_mask.pixelBuffer);
    CFRelease(inCkv.pixelBuffer);

    CFRelease(outX.pixelBuffer);
    CFRelease(outMKV.pixelBuffer);

    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif
