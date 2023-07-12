#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoderWrapper.h"
#import "CoremlDecoder.h"
#include <stdlib.h>
#import "coremlUtility.h"

// input arrays
MLMultiArray *inX;
MLMultiArray *inXa;
MLMultiArray *inQk_mask;
MLMultiArray *inMkv;
MLMultiArray *inCkv;

// output arrays
MLMultiArray *outX;
MLMultiArray *outQKs;
MLMultiArray *outMKV;
MLMultiArray *outCKV;
uint16* out_ckv_fp16;
uint16* out_qks_fp16;

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
    config.computeUnits = 3;
    const void* model = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    // input arrays
    inX = getPixelBufferArray3(5, 1, n_state);
    inXa = getPixelBufferArray3(1, 1500, n_state);
    inQk_mask = getPixelBufferArray2(1, 449);
    inMkv = getPixelBufferArray4(n_layer*2, 5, 448, n_state);
    inCkv = getPixelBufferArray4(n_layer*2, 1, 1500, n_state);

    // output arrays
    int f32_multiple = 2;
    outX = getPixelBufferArray3(5, 1, 51865 * f32_multiple);
    outMKV = getPixelBufferArray4(n_layer*2, 5, 1, n_state * f32_multiple);

    out_qks_fp16 = (uint16 *) malloc(sizeof(uint16));
    outQKs = getArray1(out_qks_fp16, 1, MLMultiArrayDataTypeFloat16);
    out_ckv_fp16 = (uint16 *) malloc(sizeof(uint16));
    outCKV = getArray1(out_ckv_fp16, 1, MLMultiArrayDataTypeFloat16);
    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* xa, // (1, 1500, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_kv_caches, // (n_layer * 2, 1, 1500, n_state)
    int n_layer,
    int n_state,
    int n_head,
    bool isNewCKV,
    float* out_x,
    float* out_cross_qks,
    float* out_new_masked_kv_caches,
    float* out_new_cross_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToFloat16(x, (uint16*)inX.dataPointer, 5 * n_state);
    if (isNewCKV) {
        float32ToFloat16(xa, (uint16*)inXa.dataPointer, 1 * 1500 * n_state);
    }
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 449);
    float32ToFloat16(masked_kv_caches, (uint16*)inMkv.dataPointer, n_layer * 2 * 5 * 448 * n_state);

    // this takes 4ms on tiny, about 40% of this func
    if (isNewCKV) {
        float32ToFloat16(cross_kv_caches, (uint16*)inCkv.dataPointer, n_layer * 2 * 1 * 1500 * n_state);
    }

    CoremlDecoderInput* input = [[CoremlDecoderInput alloc] initWithX:inX xa:inXa qk_mask:inQk_mask masked_kv_caches:inMkv cross_kv_caches:inCkv];

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
    CoremlDecoderOutput *output;

    output = (CoremlDecoderOutput*)[(__bridge id)model predictionFromFeatures:input error:&error];
    //NSLog(@"prediction %.3f", CACurrentMediaTime() - startT);

    cblas_scopy((int)   output.out_x.count,
                (float*)output.out_x.dataPointer, 1, out_x, 1);
    cblas_scopy((int)   output.out_cross_qks.count,
                (float*)output.out_cross_qks.dataPointer, 1, out_cross_qks, 1);
    cblas_scopy((int)   output.out_new_masked_kv_caches.count,
                (float*)output.out_new_masked_kv_caches.dataPointer, 1, out_new_masked_kv_caches, 1);
    cblas_scopy((int)   output.out_new_cross_kv_caches.count,
                (float*)output.out_new_cross_kv_caches.dataPointer, 1, out_new_cross_kv_caches, 1);
    if(error) {
        NSLog(@"%@", error);
    }
}

void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inX.pixelBuffer);
    CFRelease(inXa.pixelBuffer);
    CFRelease(inQk_mask.pixelBuffer);
    CFRelease(inCkv.pixelBuffer);

    CFRelease(outX.pixelBuffer);
    CFRelease(outMKV.pixelBuffer);

    free(out_qks_fp16);
    free(out_ckv_fp16);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif
