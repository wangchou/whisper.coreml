#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoderWrapper.h"
#import "CoremlDecoder.h"
#include <stdlib.h>

void float32ToFloat16(const float* fp32, uint16* fp16, int count) {
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };

    CFTimeInterval startT = CACurrentMediaTime();
    if (vImageConvert_PlanarFtoPlanar16F(&fp32Buffer, &fp16Buffer, 0) != kvImageNoError) {
        printf("float32toFloat16 error");
    }
    NSLog(@"fp32tofp16 count=%d, time %.3f", count, CACurrentMediaTime() - startT);
}

void float16ToFloat32(const uint16* fp16, float* fp32, int count) {
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    if (vImageConvert_Planar16FtoPlanarF(&fp16Buffer, &fp32Buffer, 0) != kvImageNoError) {
        printf("float16toFloat32 error");
    }
}

// in
uint16* x_fp16;
uint16* xa_fp16;
uint16* qk_mask_fp16;
uint16* mkv_fp16;
uint16* ckv_fp16;

// out
uint16* out_x_fp16;
uint16* out_qks_fp16;
uint16* out_mkv_fp16;
uint16* out_ckv_fp16;

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error = nil;
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndGPU, MLComputeUnitsAll,  MLComputeUnitsCPUAndNeuralEngine
    config.computeUnits = 3;
    const void* model = CFBridgingRetain([[CoremlDecoder alloc] initWithContentsOfURL:modelURL configuration:config error:&error]);
    if(error) {
      NSLog(@"Error load model from %s, %@", modelPath, error);
    }

    x_fp16 = (uint16 *) malloc(sizeof(uint16) * 5 * n_state);
    xa_fp16 = (uint16 *) malloc(sizeof(uint16) * 5 * 1500 * n_state);
    qk_mask_fp16 = (uint16 *) malloc(sizeof(uint16) * 449);
    mkv_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 2 * 5 * 448 * n_state);  // 1/3 of ckv_fp16
    ckv_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 2 * 5 * 1500 * n_state); // tiny~=46MB, small~=270MB, large ~= 1.2GB

    out_x_fp16 = (uint16 *) malloc(sizeof(uint16) * 5 * n_state);
    out_qks_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 5 * n_head * 1500);
    out_mkv_fp16 = (uint16 *) malloc(sizeof(uint16) * n_layer * 2 * 5 * n_state);  // 1/3 of ckv_fp16
    out_ckv_fp16 = (uint16 *) malloc(sizeof(uint16)); // tiny~=46MB, small~=270MB, large ~= 1.2GB
    return model;
}

CVPixelBufferRef getPixelBuffer(int dim1, int dim2) {
    CVPixelBufferRef pixelBuffer = NULL;
    CVReturn cvRetval = 0;
    NSDictionary* poptions = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
    cvRetval = CVPixelBufferCreate(
            kCFAllocatorDefault,
            dim1, dim2,
            kCVPixelFormatType_OneComponent16Half,
            (__bridge CFDictionaryRef)poptions,
            &pixelBuffer);

    if (cvRetval != kCVReturnSuccess) {
        NSLog(@"something wrong on creating PixelBuffer %d", cvRetval);
    }

    return pixelBuffer;
}

MLMultiArray* getPixelBufferArray2(int dim1, int dim2) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim2, dim1);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2)]];
}

MLMultiArray* getPixelBufferArray3(int dim1, int dim2, int dim3) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim3, dim1 * dim2);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3)]];
}

MLMultiArray* getPixelBufferArray4(int dim1, int dim2, int dim3, int dim4) {
    CVPixelBufferRef pixelBuffer = getPixelBuffer(dim4, dim1 * dim2 * dim3);
    return [[MLMultiArray alloc] initWithPixelBuffer:pixelBuffer shape:@[@(dim1), @(dim2), @(dim3), @(dim4)]];
}

MLMultiArray* getArray1(void* dataPtr, int dim1, MLMultiArrayDataType dataType) {
    return [[MLMultiArray alloc]
        initWithDataPointer: dataPtr
        shape: @[@(dim1)]
        dataType: dataType
        strides: @[@1]
        deallocator: nil
        error: nil
    ];
}


void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* xa, // (bs, 1500, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_kv_caches, // (n_layer * 2, bs, 1500, n_state)
    int n_layer,
    int n_state,
    int n_head,
    int text_offset,
    float* out_logits,
    float* out_cross_qks,
    float* out_new_masked_kv_caches,
    float* out_new_cross_kv_caches
) {

    NSLog(@"predictWith text_offset=%d", text_offset);

    // input arrays
    MLMultiArray *inX = getPixelBufferArray3(5, 1, n_state);
    float32ToFloat16(x, (uint16*)inX.dataPointer, 5 * n_state);

    MLMultiArray *inXa = getPixelBufferArray3(5, 1500, n_state);
    float32ToFloat16(xa, (uint16*)inXa.dataPointer, 5 * 1500 * n_state);

    MLMultiArray *inQk_mask = getPixelBufferArray2(1, 449);
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 449);

    MLMultiArray *inMkv = getPixelBufferArray4(n_layer*2, 5, 448, n_state);
    float32ToFloat16(masked_kv_caches, (uint16*)inMkv.dataPointer, n_layer * 2 * 5 * 448 * n_state);

    MLMultiArray *inCkv = getPixelBufferArray4(n_layer*2, 5, 1500, n_state);
    float32ToFloat16(cross_kv_caches, (uint16*)inCkv.dataPointer, n_layer * 2 * 5 * 1500 * n_state);

    CoremlDecoderInput* input = [[CoremlDecoderInput alloc] initWithX:inX xa:inXa qk_mask:inQk_mask masked_kv_caches:inMkv cross_kv_caches:inCkv];

    // output arrays
    MLMultiArray *outLogits =   getPixelBufferArray3(5, 1, 51865);
    MLMultiArray *outQKs = getPixelBufferArray4(n_layer*5, n_head, 1, 1500);
    MLMultiArray *outMKV = getPixelBufferArray4(n_layer*2, 5, 1, n_state);
    MLMultiArray *outCKV = getArray1(out_ckv_fp16, 1, MLMultiArrayDataTypeFloat16);
    MLPredictionOptions* options = [MLPredictionOptions alloc];

    NSDictionary *outputBackings = @{
        @"out_logits":outLogits,
        @"out_cross_qks":outQKs,
        @"out_new_masked_kv_caches":outMKV,
        @"out_new_cross_kv_caches":outCKV
    };
    [options setOutputBackings:outputBackings];

    NSError *error = nil;
    CoremlDecoderOutput *output;
    for(int i=0; i<5; i++) {
        CFTimeInterval startT = CACurrentMediaTime();
        //output = (CoremlDecoderOutput*)[(__bridge id)model predictionFromFeatures:input options:options error:&error];
        output = (CoremlDecoderOutput*)[(__bridge id)model predictionFromFeatures:input error:&error];
        NSLog(@"%d time %.3f", i, CACurrentMediaTime() - startT);

        if(error) {
            NSLog(@"%@", error);
        }
    }


   //NSLog(@"%@", output.out_x);
   // cblas_scopy((int)output.out_x.count,
   //             (float*)output.out_x.dataPointer, 1,
   //             out_x, 1);
   float16ToFloat32((uint16*)output.out_logits.dataPointer, out_logits, output.out_logits.count);
   float16ToFloat32((uint16*)output.out_cross_qks.dataPointer, out_cross_qks, output.out_cross_qks.count);
   float16ToFloat32((uint16*)output.out_new_masked_kv_caches.dataPointer, out_new_masked_kv_caches, output.out_new_masked_kv_caches.count);
   float16ToFloat32((uint16*)output.out_new_cross_kv_caches.dataPointer, out_new_cross_kv_caches, output.out_new_cross_kv_caches.count);
}

void closeModel(const void* model) {
   CFRelease(model);
   free(x_fp16);
   free(xa_fp16);
   free(qk_mask_fp16);
   free(mkv_fp16);
   free(ckv_fp16);

   free(out_x_fp16);
   free(out_qks_fp16);
   free(out_mkv_fp16);
   free(out_ckv_fp16);
}

#if __cplusplus
} //Extern C
#endif

//MLMultiArray* getArray2(void* dataPtr, int dim1, int dim2, MLMultiArrayDataType dataType) {
//    return [[MLMultiArray alloc]
//        initWithDataPointer: dataPtr
//        shape: @[@(dim1), @(dim2)]
//        dataType: dataType
//        strides: @[@(dim2), @1]
//        deallocator: nil
//        error: nil
//    ];
//}
//
//MLMultiArray* getArray3(void* dataPtr, int dim1, int dim2, int dim3, MLMultiArrayDataType dataType) {
//    return [[MLMultiArray alloc]
//        initWithDataPointer: dataPtr
//        shape: @[@(dim1), @(dim2), @(dim3)]
//        dataType: dataType
//        strides: @[@(dim2*dim3), @(dim3), @1]
//        deallocator: nil
//        error: nil
//    ];
//}
//
//MLMultiArray* getArray4(void* dataPtr, int dim1, int dim2, int dim3, int dim4, MLMultiArrayDataType dataType) {
//    return [[MLMultiArray alloc]
//        initWithDataPointer: dataPtr
//        shape: @[@(dim1), @(dim2), @(dim3), @(dim4)]
//        dataType: dataType
//        strides: @[@(dim2*dim3*dim4), @(dim3*dim4), @(dim4), @1]
//        deallocator: nil
//        error: nil
//    ];
//}
