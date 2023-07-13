#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab);
void closeModel(const void* model);
void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* xa, // (1, 1500, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_kv_caches, // (n_layer * 2, 1, 1500, n_state)
    int n_layer,
    int n_state,
    int n_head, // tiny=6, base=8, small=12, medium=16, large=20
    bool isNewCKV,
    float* out_x, // (bs, 1, n_state)
    float* out_new_masked_kv_caches // (n_layer * 2, bs, 1, n_state)
);

#if __cplusplus
}   // Extern C
#endif
