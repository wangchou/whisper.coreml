#include "decoder256Wrapper.h"
#include <stdlib.h>
#include <iostream>
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
    int n_layer = 4;
    int n_state = 384;
    int bs = 5;
    int n_head = 6; // tiny=6, base=8, small=12, medium=16, large=20
    int text_offset = 10; // only for test
    int max_n_ctx = 256;
    const void* decoder = loadModel("./tiny/CoremlDecoder256.mlmodelc", n_layer, n_state, n_head);
    // small model
    //int n_layer = 12;
    //int n_state = 768;
    //int bs = 5;
    //int n_head = 12; // tiny=6, base=8, small=12, medium=16, large=20
    //int text_offset = 10; // only for test
    //const void* decoder = loadModel("./small/CoremlDecoder256.mlmodelc", n_layer, n_state);

    float* x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* xa = getOnes(bs * 1500 * n_state); // (bs, 1500, n_state)
    float* qk_mask = getOnes(max_n_ctx * max_n_ctx); // (256, 256)

    float* out_x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* out_cross_qks = getOnes( n_layer * bs * n_head * max_n_ctx * 1500);// (n_layer * bs, n_head, 1, 1500)
    float* out_new_masked_kv_caches = getOnes( n_layer * 2 * bs * max_n_ctx * n_state); // (n_layer * 2, bs, 1, n_state)
    float* out_new_cross_kv_caches = getOnes( n_layer * 2 * 1 * 1500 * n_state); // (n_layer * 2, bs, 1, n_state)

    for(int i=0; i<5; i++) {
        predictWith(decoder, // model
                x, xa, qk_mask, // input
                n_layer, n_state, n_head, // context parameter
                out_x, out_cross_qks, out_new_masked_kv_caches, out_new_cross_kv_caches // outputs
                );
    }

    // it should match pytorch output:
    cout << " " << out_x[256*384] << " " << out_x[256*384+1] << " " << out_x[bs * 256 * 384 - 1];
    closeModel(decoder);
}