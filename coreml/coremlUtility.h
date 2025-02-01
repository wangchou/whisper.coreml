#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

void float32ToMa(const float* fp32, MLMultiArray* ma);
void maToFloat32(MLMultiArray* ma, const float* fp32);
void showStrides(MLMultiArray* ma);

MLMultiArray* getArray1(void* dataPtr, int dim1, MLMultiArrayDataType dataType);
MLMultiArray* getArray2(int dim1, int dim2);
MLMultiArray* getArray3(int dim1, int dim2, int dim3);
MLMultiArray* getArray4(int dim1, int dim2, int dim3, int dim4);

MLMultiArray* getPixelBufferArray4(int dim1, int dim2, int dim3, int dim4);
#if __cplusplus
}   // Extern C
#endif
