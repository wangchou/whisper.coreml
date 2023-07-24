#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "crossKV.h"
#import "CoremlCrossKV.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inXa;

// output arrays
MLMultiArray *outCK;
MLMultiArray *outCV;

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

    int n_head = n_state / 64;
    // input arrays
    inXa = getPixelBufferArray3(1, 1500, n_state);

    outCK = getPixelBufferArray4(n_layer, n_head, 64, 1500);
    outCV = getPixelBufferArray4(n_layer, n_head, 1500, 64);
    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* xa, // (1, 1500, n_state)
    float* out_cross_k_caches,
    float* out_cross_v_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToMa(xa, inXa);

    CoremlCrossKVInput* input = [[CoremlCrossKVInput alloc] initWithXa:inXa];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_cross_k_caches":outCK,
        @"out_cross_v_caches":outCV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlCrossKVOutput *output;

    output = (CoremlCrossKVOutput*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    maToFloat32(outCK, out_cross_k_caches);
    maToFloat32(outCV, out_cross_v_caches);

    if (!isPredicted) {
        unlock(outCK);
        unlock(outCV);
        isPredicted = true;
    }
}


void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inXa.pixelBuffer);

    CFRelease(outCK.pixelBuffer);
    CFRelease(outCV.pixelBuffer);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif

