#include "decoder256Wrapper.h"
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
    const void* decoder = loadModel("./small/CoremlDecoder256.mlmodelc", n_layer, n_state, n_head);

    float* x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* qk_mask = getOnes(max_n_ctx * max_n_ctx); // (256, 256)
    float* cross_kv_caches =  getOnes( n_layer * 2 * 1 * 1500 * n_state);// (n_layer * 2, bs, 1500, n_state)

    float* out_x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
    float* out_cross_qks = getOnes( n_layer * bs * n_head * max_n_ctx * 1500);// (n_layer * bs, n_head, 1, 1500)
    float* out_new_masked_kv_caches = getOnes( n_layer * 2 * bs * max_n_ctx * n_state); // (n_layer * 2, bs, 1, n_state)

    for(int i=0; i<5; i++) {
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        predictWith(decoder, // model
                x, qk_mask, cross_kv_caches,// input
                n_layer, n_state, n_head, i==0, // context parameter
                out_x, out_cross_qks, out_new_masked_kv_caches // outputs
                );
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Decoder256 " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    }

    // it should match pytorch output:
    cout << " " << out_x[256*384] << " " << out_x[256*384+1] << " " << out_x[bs * 256 * 384 - 1];
    closeModel(decoder);
}
