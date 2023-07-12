#if __cplusplus
extern "C" {
#endif

void loadModel(const char* modelFolderPath, int n_layer, int n_state);
void closeModel();
void predictWith(float* melSegment, float* encoderOutput);

#if __cplusplus
}   // Extern C
#endif
