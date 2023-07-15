#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state);
void closeModel(const void* model);
void predictWith(
    const void* model,
    float* xa, // (1, 1500, n_state)
    int n_layer,
    int n_state,
    float* out_cross_kv_caches // (n_layer * 2, 1, 1500, n_state)
);

#if __cplusplus
}   // Extern C
#endif
