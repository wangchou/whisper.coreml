#include <CoreFoundation/CoreFoundation.h>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>

#import "coreml.h"
#import "Decoder256.h"
#import "CrossKV.h"
#import "Encoder0.h"
#import "Decoder.h"
#import "coremlUtility.h"

#if __cplusplus
extern "C" {
#endif

// shared fp16 data array across models
MLMultiArray *arrayXa; // encoder out, crossKV in
MLMultiArray *arrayCK; // crossKV out, decoder in
MLMultiArray *arrayCV; // crossKV out, decoder in
MLMultiArray *arrayMKV448; // decoder256 outMKV256 copyed to it, and use as decoder1 input
uint16* tmpMKV[5]; // (bs, 448, n_state)

/* Encoder ------------------------------------------ */
int encoder_count;
const void* encoders[8]; // max = 32 layer / 4

MLMultiArray *inMelSegment;

bool isEncoderPredicted = false;
bool isEncoderLoaded = false;

int blockUnit = 12;
void loadEncoder(const char* modelFolderPath, int n_layer, int n_state, int n_mels) {
    encoder_count = n_layer/blockUnit;
    if (n_layer%blockUnit > 0) {
        encoder_count++;
    } // base model with layer 6

    for(int i=0; i<encoder_count; i++) {
        NSString *modelPathStr = [NSString stringWithFormat:@"%s/Encoder%d.mlmodelc", modelFolderPath, i*blockUnit]; // n blocks as sub model unit
        if (isEncoderLoaded) {
            return;
        }
        NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];
        NSError *error = nil;
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        //config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config.computeUnits = MLComputeUnitsCPUAndGPU;

        encoders[i] = CFBridgingRetain([MLModel modelWithContentsOfURL:modelURL configuration:config error:&error]);

        if(error) {
            NSLog(@"loadEncoder load model from %@, %@", modelPathStr, error);
        }
    }

    inMelSegment = getArray3(1, n_mels, 3000);
    if (arrayXa == nil) {
        arrayXa = getArray3(1, 1500, n_state);
    }

    isEncoderLoaded = true;
}

void encoderPredict(float* melSegment) {
    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
    NSDictionary *outputBackings = @{
        @"out_x":arrayXa,
    };

    [options setOutputBackings:outputBackings];

    for(int model_idx=0; model_idx < encoder_count; model_idx++) {
        MLMultiArray* inputArray;

        if(model_idx==0) {
            float32ToMa(melSegment, inMelSegment);
            inputArray = inMelSegment;
        } else {
            inputArray = arrayXa;
        }

        NSError *error = nil;

        // Encoder0Input is just a wrapper for providing interface of access
        // data by name, so it is the same for all sub encoders
        Encoder0Input* input = [[Encoder0Input alloc] initWithX:inputArray];

        [(__bridge id)encoders[model_idx] predictionFromFeatures:input options:options error:&error];
        if(error) {
            NSLog(@"encoderPredict model=%d on prediction %@", model_idx, error);
        }
    }

    if (!isEncoderPredicted) {
        isEncoderPredicted = true;
    }
}

void closeEncoder() {
    if (encoders[0] == nil) {
        return;
    }
    NSLog(@"closeEncoder");
    for(int model_idx=0; model_idx < encoder_count; model_idx++) {
        CFRelease(encoders[model_idx]);
        encoders[model_idx] = nil;
    }
    inMelSegment = nil;
    isEncoderLoaded = false;
    isEncoderPredicted = false;
}

/* CrossKV ------------------------------------------ */
bool isCrossKVPredicted = false;
bool isCrossKVLoaded = false;

const void* crossKV;

void loadCrossKV(const char* modelPath, int n_layer, int n_state) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (isCrossKVLoaded) {
        return;
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    crossKV = CFBridgingRetain([[CrossKV alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);

    if(error) {
      NSLog(@"loadCrossKV load model from %s, %@", modelPath, error);
    }

    int n_head = n_state / 64;

    if (arrayCK==nil) {
        arrayCK = getArray4(n_layer, n_head, 64, 1500);
        arrayCV = getArray4(n_layer, n_head, 1500, 64);
    }
    isCrossKVLoaded = true;
}

void crossKVPredict() {
    CrossKVInput* input = [[CrossKVInput alloc] initWithXa:arrayXa];

    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];

    NSDictionary *outputBackings = @{
        @"out_cross_k_caches":arrayCK,
        @"out_cross_v_caches":arrayCV,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    [(__bridge id)crossKV predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"crossKVPredict %@", error);
    }

    if (!isCrossKVPredicted) {
        isCrossKVPredicted = true;
    }
}

void closeCrossKV() {
    if (crossKV == nil) {
        return;
    }
    NSLog(@"closeCrossKV");
    CFRelease(crossKV);
    crossKV = nil;

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

bool isDecoder256Loaded = false;

int _n_layer;
int _n_state;
int bs = 1;

void loadDecoder256(const char* modelPath, int n_layer, int n_state, int n_head, int n_alignment_head, int decoder1_bs) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (isDecoder256Loaded) {
        return;
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    decoder256 = CFBridgingRetain([[Decoder256 alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"loadDecoder256 load model from %s, %@", modelPath, error);
    }

    int max_n_ctx = 256;

    // input arrays
    inX256 = getArray3(1, max_n_ctx, n_state);
    inQk_mask256 = getArray2(max_n_ctx, max_n_ctx);

    outX256 = getArray3(1, max_n_ctx, n_state);
    outCHW256 = getArray3(n_alignment_head, max_n_ctx, 1500);
    outMKV256 = getArray4(n_layer*2, 1, max_n_ctx, n_state);

    // prepare for decoder1 input
    _n_layer = n_layer;
    _n_state = n_state;
    bs = decoder1_bs;
    if (arrayMKV448 == nil) {
        arrayMKV448 = getPixelBufferArray4(n_layer*2, bs, 448, n_state);
    }

    // tmpMKV for rearrange_mkv
    if (tmpMKV[0] == nil) {
        for(int bi=0; bi<bs; bi++) {
            tmpMKV[bi] = (uint16*) malloc(448 * n_state * sizeof(uint16));
        }
    }

    isDecoder256Loaded = true;
}

// np_array_part = np_array[:,:,:text_offset]
// foreach layer i
//     np_array_part[i] = np_array_part[i][source_indices]
// np_array[:, :, :text_offset] = np_array_part
//
// arrayMKV448:  (n_layer * 2) * bs * 448 * n_state
void rearrange_mkv(int* indices, int text_offset) {
    int copyCount = text_offset * _n_state;
    uint16* layerPtr = (uint16*)arrayMKV448.dataPointer;

    int bsStride = 448 * _n_state;
    uint16* copyed_ptr[5];
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

void decoder256Predict(
    float* x, // (1, 256, n_state)
    float* qk_mask, // (256, 256)
    float* out_x,
    float* out_cross_head_weights,
    int beam_idx // for copy to arrayMKV448 as decoder1 input
    ) {

    // input arrays
    float32ToMa(x, inX256);
    float32ToMa(qk_mask, inQk_mask256);

    Decoder256Input* input = [[Decoder256Input alloc] initWithX:inX256 qk_mask:inQk_mask256 cross_k_caches:arrayCK cross_v_caches:arrayCV];
    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];

    NSDictionary *outputBackings = @{
        @"out_x":outX256,
        @"out_cross_head_weights":outCHW256,
        @"out_new_masked_kv_caches":outMKV256,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;

    [(__bridge id)decoder256 predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"decoder256Predict Error %@", error);
    }

    maToFloat32(outX256, out_x);

    maToFloat32(outCHW256, out_cross_head_weights);

    // arrayMKV448[:, :, :256] = outMKV256
    // arrayMKV448:  (n_layer * 2) * bs * 448 * n_state
    uint16* layer256Ptr = (uint16*)outMKV256.dataPointer;
    uint16* layer448Ptr = (uint16*)arrayMKV448.dataPointer;
    int bsStride256 = 256 * _n_state;//text_offset * _n_state;
    int bsStride448 = 448 * _n_state;
    for(int layer_i=0; layer_i < _n_layer * 2; layer_i++) {
        uint16* srcPtr = layer256Ptr;
        uint16* dstPtr = layer448Ptr + beam_idx * bsStride448;
        memcpy(dstPtr, srcPtr, bsStride256 * sizeof(uint16));

        layer256Ptr += bsStride256;
        layer448Ptr += bs * bsStride448;
    }
}

void closeDecoder256() {
    if (decoder256 == nil) {
        return;
    }
    NSLog(@"closeDecoder256");
    CFRelease(decoder256);
    decoder256 = nil;
    inX256 = nil;
    outX256 = nil;
    outCHW256 = nil;
    outMKV256 = nil;

    CFRelease(arrayMKV448.pixelBuffer);
    arrayMKV448 = nil;

    if (tmpMKV[0] != nil) {
        for(int bi=0; bi<bs; bi++) {
            CFRelease(tmpMKV[bi]);
            tmpMKV[bi] = nil;
        }
    }

    isDecoder256Loaded = false;
}

/* Decoder1 ------------------------------------------ */
const void* decoder1;

// input arrays
MLMultiArray *inX_1;
MLMultiArray *inQk_mask_1;

// output arrays
MLMultiArray *outX_1;
MLMultiArray *outMKV_1;

bool isDecoder1Loaded = false;
int _n_head;
int _n_vocab;

void loadDecoder1(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab) {
    _n_head = n_head;
    _n_vocab = n_vocab;
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    if (isDecoder1Loaded) {
        return;
    }
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndGPU;

    decoder1 = CFBridgingRetain([[Decoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);

    if(error) {
      NSLog(@"loadDecoder1 load model from %s, %@", modelPath, error);
    }

    // input arrays
    n_head = n_state/64;
    inX_1 = getArray3(bs, 1, n_state);
    if (bs == 1) {
        inQk_mask_1 = getArray2(1, 450);
    } else {
        inQk_mask_1 = getArray2(1, 449);
    }

    // output arrays
    outX_1 = getArray3(bs, 1, n_vocab);
    outMKV_1 = getArray4(n_layer*2, bs, 1, n_state);

    isDecoder1Loaded = true;
}

void decoder1Predict(
    float* x, // (bs, 1, n_state)
    float* qk_mask, // (1, 449)
    int text_offset,
    float* out_x
    ) {
    float32ToMa(x, inX_1);
    float32ToMa(qk_mask, inQk_mask_1);

    DecoderInput* input = [[DecoderInput alloc] initWithX:inX_1 qk_mask:inQk_mask_1 masked_kv_caches:arrayMKV448 cross_k_caches:arrayCK cross_v_caches:arrayCV];

    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];

    NSDictionary *outputBackings = @{
        @"out_x":outX_1,
        @"out_new_masked_kv_caches":outMKV_1,
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    [(__bridge id)decoder1 predictionFromFeatures:input options:options error:&error];

    if(error) {
        NSLog(@"decoder1Predict %@", error);
    }

    maToFloat32(outX_1, out_x);

    // mkv[:, :, text_offset] = new_mkv
    // arrayMKV448:  (n_layer * 2) * bs * 448 * n_state
    // outMKV_1: (n_layer * 2) * bs *   1 * n_state
    uint16 *dstPtr = (uint16*)arrayMKV448.dataPointer + (text_offset * _n_state);
    uint16 *srcPtr = (uint16*)outMKV_1.dataPointer;
    int dstStride = 448 * _n_state;
    int srcStride = _n_state;
    for(int i=0; i < _n_layer*2*bs; i++) {
        memcpy(dstPtr + i * dstStride,
               srcPtr + i * srcStride,
               _n_state * sizeof(uint16));
    }
}

void closeDecoder1() {
    if (decoder1 == nil) {
        return;
    }
    NSLog(@"closeDecoder1");
    CFRelease(decoder1);
    decoder1 = nil;
    inX_1 = nil;
    inQk_mask_1 = nil;
    outX_1 = nil;
    outMKV_1 = nil;

    isDecoder1Loaded = false;
}

#if __cplusplus
} //Extern C
#endif
