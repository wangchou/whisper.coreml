#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoderWrapper.h"
#import "CoremlDecoder.h"
#include <stdlib.h>

void float32ToFloat16(const float* fp32, uint16* fp16, int count) {
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    if (vImageConvert_PlanarFtoPlanar16F(&fp32Buffer, &fp16Buffer, 0) != kvImageNoError) {
        printf("float32toFloat16 error");
    }
}

void float16ToFloat32(const uint16* fp16, float* fp32, int count) {
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    if (vImageConvert_Planar16FtoPlanarF(&fp16Buffer, &fp32Buffer, 0) != kvImageNoError) {
        printf("float16toFloat32 error");
    }
}

uint16* x_fp16;
uint16* xa_fp16;
uint16* mkv_fp16;
uint16* ckv_fp16;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = 1;//MLComputeUnitsCPUAndNeuralEngine;
    const void* model = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    //const void* model = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    } //else {
      //NSLog(@"load model success");
    //}

    x_fp16 = (uint16 *) malloc(sizeof(uint16) * 5 * n_state);
    xa_fp16 = (uint16 *) malloc(sizeof(uint16) * 5 * 1500 * n_state);
    mkv_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 2 * 5 * 448 * n_state);  // 1/3 of ckv_fp16
    ckv_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 2 * 5 * 1500 * n_state); // tiny~=46MB, small~=270MB, large ~= 1.2GB
    return model;
}

void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* xa, // (bs, 1500, n_state)
    float* masked_kv_caches, // (n_layer * 2, bs, text_offset, n_state)
    float* cross_kv_caches, // (n_layer * 2, bs, 1500, n_state)
    int n_layer,
    int n_state,
    int text_offset,
    float* out_x,
    float* out_cross_qks,
    float* out_new_masked_kv_caches,
    float* out_new_cross_kv_caches
) {

    NSLog(@"predictWith text_offset=%d", text_offset);
    float32ToFloat16(x, x_fp16, 5 * n_state);
    MLMultiArray *inX = [[MLMultiArray alloc]
        initWithDataPointer: x_fp16
        shape: @[@5, @1, @(n_state)]
        dataType: MLMultiArrayDataTypeFloat16
        strides: @[@(n_state), @(n_state), @1]
        deallocator: nil
        error: nil
    ];

    float32ToFloat16(xa, xa_fp16, 5 * 1500 * n_state);
    MLMultiArray *inXa = [[MLMultiArray alloc]
        initWithDataPointer: xa_fp16
        shape: @[@5, @1500, @(n_state)]
        dataType: MLMultiArrayDataTypeFloat16
        strides: @[@(n_state*1500), @(n_state), @1]
        deallocator: nil
        error: nil
    ];

    float32ToFloat16(masked_kv_caches, mkv_fp16, n_layer * 2 * 5 * text_offset * n_state);
    MLMultiArray *inMkv = [[MLMultiArray alloc]
        initWithDataPointer: mkv_fp16
        shape: @[@(n_layer*2), @5, @(text_offset), @(n_state)]
        dataType: MLMultiArrayDataTypeFloat16
        strides: @[@(5*text_offset*n_state), @(n_state*text_offset), @(n_state), @1]
        deallocator: nil
        error: nil
    ];

    float32ToFloat16(cross_kv_caches, ckv_fp16, n_layer * 2 * 5 * 1500 * n_state);
    MLMultiArray *inCkv = [[MLMultiArray alloc]
        initWithDataPointer: ckv_fp16
        shape: @[@(n_layer*2), @5, @1500, @(n_state)]
        dataType: MLMultiArrayDataTypeFloat16
        strides: @[@(5*1500*n_state), @(n_state*1500), @(n_state), @1]
        deallocator: nil
        error: nil
    ];

    NSError *error = nil;
    CoremlDecoderOutput *output;
    for(int i=0; i<5; i++) {
        CFTimeInterval startT = CACurrentMediaTime();
        output = [(__bridge id)model predictionFromX:inX xa:inXa masked_kv_caches:inMkv cross_kv_caches:inCkv error:&error];
        NSLog(@"%d time %f", i, CACurrentMediaTime() - startT);

        if(error) {
            NSLog(@"%@", error);
        }
    }


   //NSLog(@"%@", output.out_x);
   // cblas_scopy((int)output.out_x.count,
   //             (float*)output.out_x.dataPointer, 1,
   //             out_x, 1);
   float16ToFloat32((uint16*)output.out_x.dataPointer, out_x, output.out_x.count);
   float16ToFloat32((uint16*)output.out_cross_qks.dataPointer, out_cross_qks, output.out_cross_qks.count);
   float16ToFloat32((uint16*)output.out_new_masked_kv_caches.dataPointer, out_new_masked_kv_caches, output.out_new_masked_kv_caches.count);
   float16ToFloat32((uint16*)output.out_new_cross_kv_caches.dataPointer, out_new_cross_kv_caches, output.out_new_cross_kv_caches.count);
}

void rearrangeKvCache() {}

void closeModel(const void* model) {
    CFRelease(model);
   free(x_fp16);
   free(xa_fp16);
   free(mkv_fp16);
   free(ckv_fp16);
}

#if __cplusplus
} //Extern C
#endif
