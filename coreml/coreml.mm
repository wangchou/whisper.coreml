#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>

#import "coreml.h"
#import "CoremlDecoder256.h"
#import "CoremlCrossKV.h"
#import "CoremlEncoder0.h"
#import "CoremlDecoder.h"
#import "coremlUtility.h"

#if __cplusplus
extern "C" {
#endif

/* Encoder ------------------------------------------ */
int model_count;
const void* encoders[8]; // max = 32 layer / 4

MLMultiArray *inMelSegment;
MLMultiArray *arrayXa; // encoder out, crossKV in


bool isEncoderPredicted = false;
bool isEncoderLoaded = false;

void loadEncoder(const char* modelFolderPath, int n_layer, int n_state) {
    model_count = n_layer/4;
    if (n_layer%4 > 0) {
        model_count++;
    } // base model with layer 6

    for(int i=0; i<model_count; i++) {
        CFTimeInterval startT = CACurrentMediaTime();
        NSString *modelPathStr = [NSString stringWithFormat:@"%s/CoremlEncoder%d.mlmodelc", modelFolderPath, i*4]; // 4 blocks as sub model unit
        if (!isEncoderLoaded) {
            NSLog(@"loading %@", modelPathStr);
        }
        NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];
        NSError *error = nil;
        encoders[i] = CFBridgingRetain([MLModel modelWithContentsOfURL:modelURL error:&error]);

        if(error) {
            NSLog(@"Error load model from %@, %@", modelPathStr, error);
        }

        if (!isEncoderLoaded) {
            NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
        }
    }

    inMelSegment = getPixelBufferArray3(1, 80, 3000);
    arrayXa = getPixelBufferArray3(1, 1500, n_state);

    isEncoderLoaded = true;
}

void encoderPredict(float* melSegment, float* encoderOutput) {
    MLPredictionOptions* options;
    options = [MLPredictionOptions alloc];
    NSDictionary *outputBackings = @{
        @"out_x":arrayXa,
    };

    [options setOutputBackings:outputBackings];

    int model_idx = 0;
    for(int model_idx=0; model_idx < model_count; model_idx++) {
        MLMultiArray* inputArray;

        if(model_idx==0) {
            float32ToMa(melSegment, inMelSegment);
            inputArray = inMelSegment;
        } else {
            inputArray = arrayXa;
        }

        NSError *error = nil;

        // CoremlEncoder0Input is just a wrapper for providing interface of access
        // data by name, so it is the same for all sub encoders
        CoremlEncoder0Input* input = [[CoremlEncoder0Input alloc] initWithX:inputArray];
        [(__bridge id)encoders[model_idx] predictionFromFeatures:input options:options error:&error];
        if(error) {
            NSLog(@"Error on prediction %@", error);
        }
    }

    //maToFloat32(arrayXa, encoderOutput);

    if (!isEncoderPredicted) {
        void * ptr = arrayXa.dataPointer;
        unlock(arrayXa);
        unlock(inMelSegment);
        isEncoderPredicted = true;
    }
}

void closeEncoder() {
    for(int model_idx=0; model_idx < model_count; model_idx++) {
        CFRelease(encoders[model_idx]);
    }
    CFRelease(inMelSegment.pixelBuffer);
    CFRelease(arrayXa.pixelBuffer);
    isEncoderLoaded = false;
    isEncoderPredicted = false;
}

/* CrossKV ------------------------------------------ */
MLMultiArray *arrayCK; // crossKV out, decoder in
MLMultiArray *arrayCV; // crossKV out, decoder in
bool isCrossKVPredicted = false;
bool isCrossKVLoaded = false;

const void* crossKV;

void loadCrossKV(const char* modelPath, int n_layer, int n_state) {
    CFTimeInterval startT = CACurrentMediaTime();
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (!isCrossKVLoaded) {
        NSLog(@"loading %@", modelPathStr);
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    crossKV = CFBridgingRetain([[CoremlCrossKV alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    int n_head = n_state / 64;

    arrayCK = getPixelBufferArray4(n_layer, n_head, 64, 1500);
    arrayCV = getPixelBufferArray4(n_layer, n_head, 1500, 64);
    if (!isCrossKVLoaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isCrossKVLoaded = true;
}

void crossKVPredict(
    float* xa, // (1, 1500, n_state)
    float* out_cross_k_caches,
    float* out_cross_v_caches
) {
    //CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    //if (arrayXa == nil) {
    //    float32ToMa(xa, arrayXa);
    //}

    CoremlCrossKVInput* input = [[CoremlCrossKVInput alloc] initWithXa:arrayXa];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_cross_k_caches":arrayCK,
        @"out_cross_v_caches":arrayCV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlCrossKVOutput *output;

    output = (CoremlCrossKVOutput*)[(__bridge id)crossKV predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }

    //maToFloat32(arrayCK, out_cross_k_caches);
    //maToFloat32(arrayCV, out_cross_v_caches);

    if (!isCrossKVPredicted) {
        void * ptr1 = arrayCK.dataPointer;
        void * ptr2 = arrayCV.dataPointer;
        unlock(arrayCK);
        unlock(arrayCV);
        isCrossKVPredicted = true;
    }
}

void closeCrossKV() {
    CFRelease(crossKV);

    CFRelease(arrayCK.pixelBuffer);
    CFRelease(arrayCV.pixelBuffer);
    isCrossKVLoaded = false;
    isCrossKVPredicted = false;
}

/* Decoder256 ------------------------------------------ */
const void* decoder256;

// input arrays
MLMultiArray *inX256;
MLMultiArray *inQk_mask256;

// output arrays
MLMultiArray *outX256;
MLMultiArray *outCHW256; // cross_head_weights
MLMultiArray *outMKV256;

bool isDecoder256Predicted = false;
bool isDecoder256Loaded = false;

void loadDecoder256(const char* modelPath, int n_layer, int n_state, int n_head, int n_alignment_head) {
    CFTimeInterval startT = CACurrentMediaTime();
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (!isDecoder256Loaded) {
        NSLog(@"loading %@", modelPathStr);
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    decoder256 = CFBridgingRetain([[CoremlDecoder256 alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    int max_n_ctx = 256;
    // input arrays
    inX256 = getPixelBufferArray3(1, max_n_ctx, n_state);
    inQk_mask256 = getPixelBufferArray2(max_n_ctx, max_n_ctx);

    outX256 = getPixelBufferArray3(1, max_n_ctx, n_state);
    outCHW256 = getPixelBufferArray3(n_alignment_head, max_n_ctx, 1500);
    outMKV256 = getPixelBufferArray4(n_layer*2, 1, max_n_ctx, n_state);
    if (!isDecoder256Loaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isDecoder256Loaded = true;
}

void decoder256Predict(
    float* x, // (1, 256, n_state)
    float* qk_mask, // (256, 256)
    float* cross_k_caches,
    float* cross_v_caches,
    bool isNewCKV,
    float* out_x,
    float* out_cross_head_weights,
    float* out_new_masked_kv_caches
) {
    CFTimeInterval startT = CACurrentMediaTime();

    // input arrays
    float32ToMa(x, inX256);
    float32ToMa(qk_mask, inQk_mask256);
    //if (isNewCKV) {
    //    float32ToMa(cross_k_caches, arrayCK);
    //    float32ToMa(cross_v_caches, arrayCV);
    //}

    CoremlDecoder256Input* input = [[CoremlDecoder256Input alloc] initWithX:inX256 qk_mask:inQk_mask256 cross_k_caches:arrayCK cross_v_caches:arrayCV];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_x":outX256,
        @"out_cross_head_weights":outCHW256,
        @"out_new_masked_kv_caches":outMKV256,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoder256Output *output;

    output = (CoremlDecoder256Output*)[(__bridge id)decoder256 predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"Decoder256 Error %@", error);
    }

    maToFloat32(outX256, out_x);

    maToFloat32(outCHW256, out_cross_head_weights);

    maToFloat32(outMKV256, out_new_masked_kv_caches);
    if (!isDecoder256Predicted) {
        unlock(outX256);
        unlock(outCHW256);
        unlock(outMKV256);
        isDecoder256Predicted = true;
    }
}

void closeDecoder256() {
    CFRelease(decoder256);
    CFRelease(inX256.pixelBuffer);

    CFRelease(outX256.pixelBuffer);
    CFRelease(outCHW256.pixelBuffer);
    CFRelease(outMKV256.pixelBuffer);
    isDecoder256Loaded = false;
    isDecoder256Predicted = false;
}

/* Decoder ------------------------------------------ */
const void* decoder1;

// input arrays
MLMultiArray *inX_1;
MLMultiArray *inQk_mask_1;
MLMultiArray *inMkv_1;

// output arrays
MLMultiArray *outX_1;
MLMultiArray *outMKV_1;

bool isDecoder1Predicted = false;
bool isDecoder1Loaded = false;
int _n_layer;
int _n_state;
int _n_head;
int _n_vocab;
int bs = 1;

uint16* tmpMKV[5]; // (bs, 448, n_state)

void loadDecoder1(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab, int beam_size) {
    CFTimeInterval startT = CACurrentMediaTime();
    _n_layer = n_layer;
    _n_state = n_state;
    _n_head = n_head;
    _n_vocab = n_vocab;
    bs = beam_size;
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (!isDecoder1Loaded) {
        NSLog(@"loading %@", modelPathStr);
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    decoder1 = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    // input arrays
    n_head = n_state/64;
    inX_1 = getPixelBufferArray3(bs, 1, n_state);
    if (beam_size == 1) {
        inQk_mask_1 = getPixelBufferArray2(1, 450);
    } else {
        inQk_mask_1 = getPixelBufferArray2(1, 449);
    }
    inMkv_1 = getPixelBufferArray4(n_layer*2, bs, 448, n_state);

    // output arrays
    outX_1 = getPixelBufferArray3(bs, 1, n_vocab);
    outMKV_1 = getPixelBufferArray4(n_layer*2, bs, 1, n_state);

    // tmpMKV for rearrange_mkv
    for(int bi=0; bi<bs; bi++) {
        tmpMKV[bi] = (uint16*) malloc(448 * n_state * sizeof(uint16));
    }

    if (!isDecoder1Loaded) {
        NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
    }
    isDecoder1Loaded = true;
}

// np_array_part = np_array[:,:,:text_offset]
// foreach layer i
//     np_array_part[i] = np_array_part[i][source_indices]
// np_array[:, :, :text_offset] = np_array_part
//
// inMkv_1:  (n_layer * 2) * 5 * 448 * n_state
uint16* copyed_ptr[5];
void rearrange_mkv(int* indices, int text_offset) {
    //NSLog(@"objc rearrange_mkv indices=%d,%d,%d,%d,%d... text_offset=%d", indices[0], indices[1],indices[2], indices[3], indices[4], text_offset);
    int copyCount = 448 * _n_state;//text_offset * _n_state;
    uint16* layerPtr = (uint16*)inMkv_1.dataPointer;

    int bsStride = 448 * _n_state;
    for(int layer_i=0; layer_i < _n_layer * 2; layer_i++) {
        // copy to tmp buffer
        for(int bi=0; bi<bs; bi++) {
            uint16* srcPtr = layerPtr + bi * bsStride;
            uint16* dstPtr = indices[bi] == bi ? srcPtr : tmpMKV[bi];
            if (srcPtr != dstPtr) {
                memcpy(dstPtr, srcPtr, copyCount * sizeof(uint16));
            }
            copyed_ptr[bi] = dstPtr;
        }
        // copy from tmpBuffer back to origin
        for(int bi=0; bi<bs; bi++) {
            uint16* srcPtr = copyed_ptr[indices[bi]];
            uint16* dstPtr = layerPtr + bi * bsStride;
            if (srcPtr != dstPtr) {
                memcpy(dstPtr, srcPtr, copyCount * sizeof(uint16));
            }
        }
        layerPtr += bs * 448 * _n_state;
    }
}

void decoder1Predict(
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
    float32ToMa(x, inX_1);
    float32ToMa(qk_mask, inQk_mask_1);

    if (isNewCKV) {
        float32ToMa(masked_kv_caches, inMkv_1);
        //if (arrayCK == nil) {
        //    float32ToMa(cross_k_caches, arrayCK);
        //    float32ToMa(cross_v_caches, arrayCV);
        //}
    }
    //NSLog(@"\tinput fp32->fp16  %.4f", CACurrentMediaTime() - startT);
    //startT = CACurrentMediaTime();

    CoremlDecoderInput* input = [[CoremlDecoderInput alloc] initWithX:inX_1 qk_mask:inQk_mask_1 masked_kv_caches:inMkv_1 cross_k_caches:arrayCK cross_v_caches:arrayCV];

    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_x":outX_1,
        @"out_new_masked_kv_caches":outMKV_1,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoderOutput *output;

    output = (CoremlDecoderOutput*)[(__bridge id)decoder1 predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"%@", error);
    }
    //NSLog(@"\tpredict           %.4f", CACurrentMediaTime() - startT);
    //startT = CACurrentMediaTime();
    maToFloat32(outX_1, out_x);

    maToFloat32(outMKV_1, out_new_masked_kv_caches);

    // mkv[:, :, text_offset] = new_mkv
    // inMkv_1:  (n_layer * 2) * bs * 448 * n_state
    // outMKV_1: (n_layer * 2) * bs *   1 * n_state
    uint16 *dstPtr = (uint16*)inMkv_1.dataPointer + (text_offset * _n_state);
    uint16 *srcPtr = (uint16*)outMKV_1.dataPointer;
    int dstStride = 448 * _n_state;
    int srcStride = _n_state;
    for(int i=0; i < _n_layer*2*bs; i++) {
        memcpy(dstPtr + i * dstStride,
               srcPtr + i * srcStride,
               _n_state * sizeof(uint16));
    }
    //NSLog(@"\toutput fp16->fp32 %.4f", CACurrentMediaTime() - startT);

    if (!isDecoder1Predicted) {
        unlock(inMkv_1);
        unlock(outX_1);
        unlock(outMKV_1);
        isDecoder1Predicted = true;
    }
}

void closeDecoder1() {
    CFRelease(decoder1);
    CFRelease(inX_1.pixelBuffer);
    CFRelease(inQk_mask_1.pixelBuffer);

    CFRelease(outX_1.pixelBuffer);
    CFRelease(outMKV_1.pixelBuffer);
    for(int i=0; i<bs; i++) {
        free(tmpMKV[i]);
    }

    isDecoder1Loaded = false;
    isDecoder1Predicted = false;
}

#if __cplusplus
} //Extern C
#endif
