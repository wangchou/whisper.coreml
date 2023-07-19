#import <CoreML/CoreML.h>
#include <malloc/_malloc.h>
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
MLMultiArray *inCk;
MLMultiArray *inCv;

// output arrays
MLMultiArray *outX;
MLMultiArray *outMKV;

bool isPredicted = false;
bool isModelLoaded = false;
int _n_layer;
int _n_state;
int _n_head;
int _n_vocab;

uint16* tmpMKV[5]; // (5, 448, n_state)

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab) {
    CFTimeInterval startT = CACurrentMediaTime();
    _n_layer = n_layer;
    _n_state = n_state;
    _n_head = n_head;
    _n_vocab = n_vocab;
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
    n_head = n_state/64;
    inX = getPixelBufferArray3(5, 1, n_state);
    inQk_mask = getPixelBufferArray2(1, 449);
    inMkv = getPixelBufferArray4(n_layer*2, 5, 448, n_state);
    inCk = getPixelBufferArray4(n_layer, n_head, 64, 1500);
    inCv = getPixelBufferArray4(n_layer, n_head, 1500, 64);

    // output arrays
    outX = getPixelBufferArray3(5, 1, n_vocab);
    outMKV = getPixelBufferArray4(n_layer*2, 5, 1, n_state);

    // tmpMKV for rearrange_mkv
    for(int bs=0; bs<5; bs++) {
        tmpMKV[bs] = (uint16*) malloc(448 * n_state * sizeof(uint16));
    }

    if (!isModelLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isModelLoaded = true;
    return model;
}

// np_array_part = np_array[:,:,:text_offset]
// foreach layer i
//     np_array_part[i] = np_array_part[i][source_indices]
// np_array[:, :, :text_offset] = np_array_part
//
// inMkv:  (n_layer * 2) * 5 * 448 * n_state
uint16* copyed_ptr[5];
void rearrange_mkv(int* indices, int text_offset) {
    //NSLog(@"objc rearrange_mkv indices=%d,%d,%d,%d,%d... text_offset=%d", indices[0], indices[1],indices[2], indices[3], indices[4], text_offset);
    int copyCount = 448 * _n_state;//text_offset * _n_state;
    uint16* layerPtr = (uint16*)inMkv.dataPointer;

    int bsStride = 448 * _n_state;
    for(int layer_i=0; layer_i < _n_layer * 2; layer_i++) {
        // copy to tmp buffer
        for(int bs=0; bs<5; bs++) {
            uint16* srcPtr = layerPtr + bs * bsStride;
            uint16* dstPtr = indices[bs] == bs ? srcPtr : tmpMKV[bs];
            if (srcPtr != dstPtr) {
                memcpy(dstPtr, srcPtr, copyCount * sizeof(uint16));
            }
            copyed_ptr[bs] = dstPtr;
        }
        // copy from tmpBuffer back to origin
        for(int bs=0; bs<5; bs++) {
            uint16* srcPtr = copyed_ptr[indices[bs]];
            uint16* dstPtr = layerPtr + bs * bsStride;
            if (srcPtr != dstPtr) {
                memcpy(dstPtr, srcPtr, copyCount * sizeof(uint16));
            }
        }
        layerPtr += 5 * 448 * _n_state;
    }
}

void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_k_caches,
    float* cross_v_caches,
    int text_offset,
    bool isNewCKV,
    float* out_x,
    float* out_new_masked_kv_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToMa(x, inX);
    float32ToMa(qk_mask, inQk_mask);

    if (isNewCKV) {
        float32ToMa(masked_kv_caches, inMkv);
        float32ToMa(cross_k_caches, inCk);
        float32ToMa(cross_v_caches, inCv);
    }
    //NSLog(@"\tinput fp32->fp16  %.4f", CACurrentMediaTime() - startT);
    //startT = CACurrentMediaTime();

    CoremlDecoderInput* input = [[CoremlDecoderInput alloc] initWithX:inX qk_mask:inQk_mask masked_kv_caches:inMkv cross_k_caches:inCk cross_v_caches:inCv];

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
    //NSLog(@"\tpredict           %.4f", CACurrentMediaTime() - startT);
    //startT = CACurrentMediaTime();
    maToFloat32(outX, out_x);

    maToFloat32(outMKV, out_new_masked_kv_caches);

    // mkv[:, :, text_offset] = new_mkv
    // inMkv:  (n_layer * 2) * 5 * 448 * n_state
    // outMKV: (n_layer * 2) * 5 *   1 * n_state
    uint16 *dstPtr = (uint16*)inMkv.dataPointer + (text_offset * _n_state);
    uint16 *srcPtr = (uint16*)outMKV.dataPointer;
    int dstStride = 448 * _n_state;
    int srcStride = _n_state;
    for(int i=0; i < _n_layer*2*5; i++) {
        memcpy(dstPtr + i * dstStride,
               srcPtr + i * srcStride,
               _n_state * sizeof(uint16));
    }
    //NSLog(@"\toutput fp16->fp32 %.4f", CACurrentMediaTime() - startT);

    if (!isPredicted) {
        unlock(inMkv);
        unlock(outX);
        unlock(outMKV);
        isPredicted = true;
    }
}

void closeModel(const void* model) {
    CFRelease(model);
    CFRelease(inX.pixelBuffer);
    CFRelease(inQk_mask.pixelBuffer);
    CFRelease(inCk.pixelBuffer);
    CFRelease(inCv.pixelBuffer);

    CFRelease(outX.pixelBuffer);
    CFRelease(outMKV.pixelBuffer);
    for(int i=0; i<5; i++) {
        free(tmpMKV[i]);
    }

    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif
