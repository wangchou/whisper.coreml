#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "decoderWrapper.h"
#import "CoremlDecoder.h"
#include <stdlib.h>

void float32ToFloat16(const float* fp32, uint16* fp16, int count) {
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };

    //CFTimeInterval startT = CACurrentMediaTime();
    if (vImageConvert_PlanarFtoPlanar16F(&fp32Buffer, &fp16Buffer, 0) != kvImageNoError) {
        printf("float32toFloat16 error");
    }
    //NSLog(@"fp32tofp16 count=%d, time %.3f", count, CACurrentMediaTime() - startT);
}

void float16ToFloat32(const uint16* fp16, float* fp32, int count) {
    vImage_Buffer fp16Buffer = { (void *)fp16, 1, UInt(count), count * 2 };
    vImage_Buffer fp32Buffer = { (void *)fp32, 1, UInt(count), count * 4 };
    if (vImageConvert_Planar16FtoPlanarF(&fp16Buffer, &fp32Buffer, 0) != kvImageNoError) {
        printf("float16toFloat32 error");
    }
}

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

#if __cplusplus
extern "C" {
#endif

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

    // input arrays
    inX = getPixelBufferArray3(5, 1, n_state);
    inXa = getPixelBufferArray3(5, 1500, n_state);
    inQk_mask = getPixelBufferArray2(1, 449);
    inMkv = getPixelBufferArray4(n_layer*2, 5, 448, n_state);
    inCkv = getPixelBufferArray4(n_layer*2, 5, 1500, n_state);

    // output arrays
    int f32_multiple = 2;
    outX = getPixelBufferArray3(5, 1, 51865 * f32_multiple);
    outQKs = getPixelBufferArray4(n_layer*5, n_head, 1, 1500 * f32_multiple);
    outMKV = getPixelBufferArray4(n_layer*2, 5, 1, n_state * f32_multiple);
    outCKV = getArray1(out_ckv_fp16, 1, MLMultiArrayDataTypeFloat16);

    out_ckv_fp16 = (uint16 *) malloc(sizeof(uint16)); // tiny~=46MB, small~=270MB, large ~= 1.2GB
    outCKV = getArray1(out_ckv_fp16, 1, MLMultiArrayDataTypeFloat16);
    return model;
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
        float32ToFloat16(xa, (uint16*)inXa.dataPointer, 5 * 1500 * n_state);
    }
    float32ToFloat16(qk_mask, (uint16*)inQk_mask.dataPointer, 449);
    float32ToFloat16(masked_kv_caches, (uint16*)inMkv.dataPointer, n_layer * 2 * 5 * 448 * n_state);

    // this takes 4ms on tiny, about 40% of this func
    if (isNewCKV) {
        float32ToFloat16(cross_kv_caches, (uint16*)inCkv.dataPointer, n_layer * 2 * 5 * 1500 * n_state);
    }

    //NSLog(@"f32 to f16 %.3f %s", CACurrentMediaTime() - startT, isNewCKV ? "True" : "False");

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
   CFRelease(outQKs.pixelBuffer);
   CFRelease(outMKV.pixelBuffer);

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
