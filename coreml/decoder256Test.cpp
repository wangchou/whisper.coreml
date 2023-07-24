#include "decoder256.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>
using namespace std;

float* getOnes(int count) {
    float* ptr = (float *)malloc(sizeof(float) * count);
    for(int i=0; i < count; i++) {
        ptr[i] = 1.0;
    }
    return ptr;
}

int main() {
    // tiny model
    //int n_layer = 4;
    //int n_state = 384;
    //int bs = 1;
    //int n_head = 6; // tiny=6, base=8, small=12, medium=16, large=20
    //int text_offset = 10; // only for test
    //int max_n_ctx = 256;
    //const void* decoder = loadModel("./tiny/CoremlDecoder256.mlmodelc", n_layer, n_state, n_head);
    // small model
    int n_layer = 12;
    int n_state = 768;
    int bs = 1;
    int n_head = 12; // tiny=6, base=8, small=12, medium=16, large=20
    int text_offset = 10; // only for test
    int max_n_ctx = 256;
    int n_alignment_head = 10;
    const void* decoder = loadModel("./small/CoremlDecoder256.mlmodelc", n_layer, n_state, n_head, n_alignment_head);

    // memory usage on large model
    // fp32 cross_kv_caches 491MB (32 * 2 * 1500 * 1280 * 4 bytes)
    // fp32 out_cross_head_weights 15MB (10 * 256 * 1500 * 4 bytes)
    // fp32 out_new_masked_kv_caches 83MB
    // fp32 ~= 600MB
    //
    // fp16 CVPixelBuffer ~= fp32/2 ~= 300MB
    // total = 0.9GB + ane load model (?GB, ps: static full model = 1.47GB)
    //       => 0.9GB ~ 2.3+GB

    float* x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* qk_mask = getOnes(max_n_ctx * max_n_ctx); // (256, 256)
    float* cross_k_caches = getOnes(n_layer * 1500 * n_state);
    float* cross_v_caches = getOnes(n_layer * 1500 * n_state);

    float* out_x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* out_cross_head_weights = getOnes(n_alignment_head * max_n_ctx * 1500);
    float* out_new_masked_kv_caches = getOnes(n_layer * 2 * bs * max_n_ctx * n_state); // (n_layer * 2, bs, 256, n_state)

    for(int i=0; i<5; i++) {
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        predictWith(decoder, // model
                x, qk_mask, cross_k_caches, cross_v_caches,// input
                i==0, // context parameter
                out_x, out_cross_head_weights, out_new_masked_kv_caches // outputs
                );
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Decoder256 " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    }

    // it should match pytorch output:
    cout << " " << out_x[256*384] << " " << out_x[256*384+1] << " " << out_x[bs * 256 * 384 - 1];
    closeModel(decoder);
}
    //n_alignment_head = {"tiny.en": 8,
    //    "tiny": 6,
    //    "base.en": ,
    //    "base": 8,
    //    "small.en": 19,
    //    "small": 10,
    //    "medium.en": ,
    //    "medium": 6,
    //    "large-v1": ,
    //    "large-v2": 23,
    //    "large": 23,
    //}

    //// large model
    //int n_layer = 32;
    //int n_state = 1280;
    //int bs = 1;
    //int n_head = 20; // tiny=6, base=8, small=12, medium=16, large=20
    //int text_offset = 10; // only for test
    //int max_n_ctx = 256;
    //const void* decoder = loadModel("./large/CoremlDecoder256.mlmodelc", n_layer, n_state, n_head);
