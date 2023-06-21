#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath);
void closeModel(const void* model);
void predictWith(const void* model, float* melSegment, float* encoderOutput);

#if __cplusplus
}   // Extern C
#endif
