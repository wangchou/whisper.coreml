#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab);
void closeModel(const void* model);
void rearrange_mkv(int* indices, int text_offset);
void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* qk_mask, // (1, 449)
    float* masked_kv_caches, // (n_layer * 2, bs, 448, n_state)
    float* cross_k_caches, // (n_layer, n_head, 64, 1500)
    float* cross_v_caches, // (n_layer, n_head, 1500, 64)
    int text_offset,
    bool isNewCKV,
    float* out_x, // (bs, 1, n_state)
    float* out_new_masked_kv_caches // (n_layer * 2, bs, 1, n_state)
);

#if __cplusplus
}   // Extern C
#endif
