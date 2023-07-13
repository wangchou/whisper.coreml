#import <CoreML/CoreML.h>
#include <malloc/_malloc.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#import "encoderWrapper.h"
#import "coremlUtility.h"
#import "CoremlEncoder0.h"
#include <stdlib.h>

MLMultiArray *arrayX;
MLMultiArray *arrayMelSegment;
MLPredictionOptions* options;
const void* models[8]; // max = 32 layer / 4
int model_count;
bool isPredicted = false;
bool isModelLoaded = false;

#if __cplusplus
extern "C" {
#endif

void loadModel(const char* modelFolderPath, int n_layer, int n_state) {
    model_count = n_layer/4;
    if (n_layer%4 > 0) {
        model_count++;
    } // base model with layer 6

    if (!isModelLoaded) {
        NSLog(@"load encoder, submodel_count=%d", model_count);
    }

    for(int i=0; i<model_count; i++) {
        CFTimeInterval startT = CACurrentMediaTime();
        NSString *modelPathStr = [NSString stringWithFormat:@"%s/CoremlEncoder%d.mlmodelc", modelFolderPath, i*4]; // 4 blocks as sub model unit
        if (!isModelLoaded) {
            NSLog(@"loading %@", modelPathStr);
        }
        NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];
        NSError *error = nil;
        models[i] = CFBridgingRetain([MLModel modelWithContentsOfURL:modelURL error:&error]);

        if(error) {
            NSLog(@"Error load model from %@, %@", modelPathStr, error);
        }

        if (!isModelLoaded) {
            NSLog(@"loaded in %.3fs", CACurrentMediaTime() - startT);
        }
    }

    arrayMelSegment = getPixelBufferArray3(1, 80, 3000);
    arrayX = getPixelBufferArray3(1, 1500, n_state);

    options = [MLPredictionOptions alloc];
    NSDictionary *outputBackings = @{
        @"out_x":arrayX,
    };

    [options setOutputBackings:outputBackings];
    isModelLoaded = true;
}

void predictWith(float* melSegment, float* encoderOutput) {
    int model_idx = 0;
    for(int model_idx=0; model_idx < model_count; model_idx++) {
        MLMultiArray* inputArray;

        //showStrides(arrayMelSegment);
        if(model_idx==0) {
            // fp32 to fp16
            float* fromPtr = melSegment;
            uint16* toPtr = (uint16*)arrayMelSegment.dataPointer;
            for(int i=0; i<80; i++) {
                float32ToFloat16(fromPtr, toPtr, 3000);
                fromPtr += 3000;
                toPtr += 3008; // ane last dim are 64bytes aligned
            }
            inputArray = arrayMelSegment;
        } else {
            inputArray = arrayX;
        }

        NSError *error = nil;

        // CoremlEncoder0Input is just a wrapper for providing interface of access
        // data by name, so it is the same for all sub models
        CoremlEncoder0Input* input = [[CoremlEncoder0Input alloc] initWithX:inputArray];
        [(__bridge id)models[model_idx] predictionFromFeatures:input options:options error:&error];
        if(error) {
            NSLog(@"Error on prediction %@", error);
        }
    }

    float16ToFloat32((uint16*)arrayX.dataPointer, encoderOutput, arrayX.count);
    if (!isPredicted) {
        unlock(arrayX);
        unlock(arrayMelSegment);
        isPredicted = true;
    }
}

void closeModel() {
    for(int model_idx=0; model_idx < model_count; model_idx++) {
        CFRelease(models[model_idx]);
    }
    //Release(options);
    CFRelease(arrayMelSegment.pixelBuffer);
    CFRelease(arrayX.pixelBuffer);
    isModelLoaded = false;
    isPredicted = false;
}

#if __cplusplus
} //Extern C
#endif
