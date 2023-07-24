#if __cplusplus
extern "C" {
#endif

void loadEncoder(const char* modelFolderPath, int n_layer, int n_state);
void closeEncoder();
void encoderPredict(float* melSegment);

void loadCrossKV(const char* modelPath, int n_layer, int n_state);
void closeCrossKV();
void crossKVPredict();

void loadDecoder256(const char* modelPath, int n_layer, int n_state, int n_head, int n_alignment_head, int beam_size);
void closeDecoder256();
void decoder256Predict(
    float* x, // (1, 256, n_state)
    float* qk_mask, // (256, 256)
    float* out_x, // (1, 256, n_state)
    float* out_cross_head_weights, // (n__alignment_head, 256, 1500)
    int beam_idx
);

void loadDecoder1(const char* modelPath, int n_layer, int n_state, int n_head, int n_vocab);
void closeDecoder1();
void rearrange_mkv(int* indices, int text_offset);
void decoder1Predict(
    float* x, // (bs, 1, n_state)
    float* qk_mask, // (1, 449)
    int text_offset,
    float* out_x, // (bs, 1, n_state)
    float* out_new_masked_kv_caches // (n_layer * 2, bs, 1, n_state)
);

#if __cplusplus
}   // Extern C
#endif
