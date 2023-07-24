#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state);
void closeModel(const void* model);
void predictWith(
    const void* model,
    float* xa, // (1, 1500, n_state)
    float* out_cross_k_caches, // (n_layer, n_head, 64, 1500)
    float* out_cross_v_caches // (n_layer, n_head, 1500, 64)
);

#if __cplusplus
}   // Extern C
#endif
