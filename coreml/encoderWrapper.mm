#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "encoderWrapper.h"
#import "CoremlEncoder.h"
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    const void* model = CFBridgingRetain([[CoremlEncoder alloc] initWithContentsOfURL:modelURL error:nil]);
    return model;
}

void predictWith(const void* model, float* melSegment, float* encoderOutput) {
    MLMultiArray *inMultiArray = [[MLMultiArray alloc] initWithDataPointer: melSegment
                                                                      shape: @[@1, @80, @3000]
                                                                   dataType: MLMultiArrayDataTypeFloat32
                                                                    strides: @[@(240000), @(3000), @1]
                                                                deallocator: nil
                                                                      error: nil];

    CoremlEncoderOutput *modelOutput = [(__bridge id)model predictionFromMelSegment:inMultiArray error:nil];
    MLMultiArray *outMA = modelOutput.output;

    cblas_scopy((int)outMA.count,
                (float*)outMA.dataPointer, 1,
                encoderOutput, 1);
}

void closeModel(const void* model) {
    CFRelease(model);
}

#if __cplusplus
} //Extern C
#endif
