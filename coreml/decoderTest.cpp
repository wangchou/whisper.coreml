#include "decoderWrapper.h"
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
                          //
    const void* decoder = loadModel("./tiny/CoremlDecoder.mlmodelc", n_layer, n_state);

    float* x = getOnes(bs * n_state); // (bs, 1, n_state)
    float* xa = getOnes(bs * 1500 * n_state); // (bs, 1500, n_state)
    float* masked_kv_caches = getOnes( n_layer * 2 * bs * text_offset * n_state); // (n_layer * 2, bs, text_offset, n_state)
    float* cross_kv_caches =  getOnes( n_layer * 2 * bs * 1500 * n_state);// (n_layer * 2, bs, 1500, n_state)

    float* out_x = getOnes(bs * n_state); // (bs, 1, n_state)
    float* out_cross_qks = getOnes( n_layer * 2 * bs * n_head * 1500);// (n_layer * bs, n_head, 1, 1500)
    float* out_new_masked_kv_caches = getOnes( n_layer * 2 * bs * 1 * n_state); // (n_layer * 2, bs, 1, n_state)
    float* out_new_cross_kv_caches  = getOnes( 1);// (1)

    predictWith(decoder, // model
                x, xa, masked_kv_caches, cross_kv_caches, // input
                n_layer, n_state, text_offset, // context parameter
                out_x, out_cross_qks, out_new_masked_kv_caches, out_new_cross_kv_caches // outputs
                );

    // it should match pytorch output:
    // tensor([ 0.9057, -1.3382], grad_fn=<SliceBackward0>) tensor(7.6636, grad_fn=<SelectBackward0>)
    cout << " " << out_x[0] << " " << out_x[1] << " " << out_x[bs * n_state - 1];
    closeModel(decoder);
}
