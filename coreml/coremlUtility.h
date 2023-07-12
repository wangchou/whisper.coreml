#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

void float32ToFloat16(const float* fp32, uint16* fp16, int count);
void float16ToFloat32(const uint16* fp16, float* fp32, int count);
CVPixelBufferRef getPixelBuffer(int dim1, int dim2);
MLMultiArray* getPixelBufferArray2(int dim1, int dim2);
MLMultiArray* getPixelBufferArray3(int dim1, int dim2, int dim3);
MLMultiArray* getPixelBufferArray4(int dim1, int dim2, int dim3, int dim4);
MLMultiArray* getArray1(void* dataPtr, int dim1, MLMultiArrayDataType dataType);

#if __cplusplus
}   // Extern C
#endif
