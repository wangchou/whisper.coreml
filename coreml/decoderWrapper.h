#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath, int n_layer, int n_state);
void closeModel(const void* model);
void predictWith(
    const void* model,
    float* x, // (bs, 1, n_state)
    float* xa, // (bs, 1500, n_state)
    float* masked_kv_caches, // (n_layer * 2, bs, text_offset, n_state)
    float* cross_kv_caches, // (n_layer * 2, bs, 1500, n_state)
    int n_layer,
    int n_state,
    int text_offset,
    //int n_head, // tiny=6, base=8, small=12, medium=16, large=20
    float* out_x, // (bs, 1, n_state)
    float* out_cross_qks, // (n_layer * bs, n_head, 1, 1500)
    float* out_new_masked_kv_caches, // (n_layer * 2, bs, 1, n_state)
    float* out_new_cross_kv_caches // (1)
);

#if __cplusplus
}   // Extern C
#endif
