#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "crossKVWrapper.h"
#import "CoremlCrossKV.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inXa;

// output arrays
MLMultiArray *outCKV;

bool isPredicted = false;
bool isModelLoaded = false;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state) {
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
    const void* model = CFBridgingRetain([[CoremlCrossKV alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    // input arrays
    inXa = getPixelBufferArray3(1, 1500, n_state);

    outCKV = getPixelBufferArray4(n_layer*2, 1, 1500, n_state);
    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* xa, // (1, 1500, n_state)
    int n_layer,
    int n_state,
    float* out_cross_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToFloat16(xa, (uint16*)inXa.dataPointer, 1 * 1500 * n_state);

    CoremlCrossKVInput* input = [[CoremlCrossKVInput alloc] initWithXa:inXa];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_cross_kv_caches":outCKV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlCrossKVOutput *output;

    output = (CoremlCrossKVOutput*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    float16ToFloat32((uint16*)outCKV.dataPointer, out_cross_kv_caches, outCKV.count);

    if (!isPredicted) {
        unlock(outCKV);
        isPredicted = true;
    }
}


void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inXa.pixelBuffer);

    CFRelease(outCKV.pixelBuffer);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif

