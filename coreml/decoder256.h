#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_alignment_head);
void closeModel(const void* model);
void predictWith(
    const void* model,
    float* x, // (1, 256, n_state)
    float* qk_mask, // (256, 256)
    float* cross_k_caches, // (n_layer, n_head, 64, 1500)
    float* cross_v_caches, // (n_layer, n_head, 1500, 64)
    bool isNewCKV,
    float* out_x, // (1, 256, n_state)
    float* out_cross_head_weights, // (n__alignment_head, 256, 1500)
    float* out_new_masked_kv_caches // (n_layer * 2, 1, 256, n_state)
);

#if __cplusplus
}   // Extern C
#endif
