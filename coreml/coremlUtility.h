#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

void float32ToMa(const float* fp32, MLMultiArray* ma);
void maToFloat32(MLMultiArray* ma, const float* fp32);
void unlock(MLMultiArray* ma);
void showStrides(MLMultiArray* ma);
CVPixelBufferRef getPixelBuffer(int dim1, int dim2);
MLMultiArray* getArray2(int dim1, int dim2);
MLMultiArray* getArray3(int dim1, int dim2, int dim3);
MLMultiArray* getArray4(int dim1, int dim2, int dim3, int dim4);
MLMultiArray* getArray1(void* dataPtr, int dim1, MLMultiArrayDataType dataType);

#if __cplusplus
}   // Extern C
#endif
