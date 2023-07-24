#include "decoder.h"
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
    //int bs = 5;
    //int n_head = 6; // tiny=6, base=8, small=12, medium=16, large=20
    //int text_offset = 10; // only for test
    //const void* decoder = loadModel("./tiny/CoremlDecoder.mlmodelc", n_layer, n_state, n_head);

    // base model
    //int n_layer = 6;
    //int n_state = 512;
    //int bs = 5;
    //int n_head = 8; // tiny=6, base=8, small=12, medium=16, large=20
    //int text_offset = 10; // only for test
    //int n_vocab = 51864; //multi-lang: 51865, en only: 51864
    //const void* decoder = loadModel("./base.en/CoremlDecoder.mlmodelc", n_layer, n_state, n_head, n_vocab);

    // small model
    int n_layer = 12;
    int n_state = 768;
    int bs = 1;
    int n_head = 12; // tiny=6, base=8, small=12, medium=16, large=20
    int text_offset = 10; // only for test
    int n_vocab = 51865; //multi-lang: 51865, en only: 51864
    const void* decoder = loadModel("./small/CoremlDecoder.mlmodelc", n_layer, n_state, n_head, n_vocab, bs);

    float* x = getOnes(bs * n_state); // (bs, 1, n_state)
    float* qk_mask = getOnes(450); // (1, 449)
    float* masked_kv_caches = getOnes( n_layer * 2 * bs * 448 * n_state); // (n_layer * 2, 1, 448, n_state)
    float* cross_k_caches =  getOnes( n_layer * 1 * 1500 * n_state);
    float* cross_v_caches =  getOnes( n_layer * 1 * 1500 * n_state);

    float* out_x = getOnes(bs * n_vocab); // (bs, 1, n_state)
    float* out_new_masked_kv_caches = getOnes( n_layer * 2 * bs * 1 * n_state); // (n_layer * 2, bs, 1, n_state)

    for(int i=0; i<5; i++) {
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        predictWith(decoder, // model
                x, qk_mask, masked_kv_caches, cross_k_caches, cross_v_caches,// input
                text_offset, i==0, // context parameter
                out_x, out_new_masked_kv_caches // outputs
                );
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "decoder1 " << chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << ", isNewCKV=" << (i==0) << endl;
        cout << "---" << endl;
    }

    // it should match pytorch output:
    // tensor([ 0.9057, -1.3382], grad_fn=<SliceBackward0>) tensor(7.6636, grad_fn=<SelectBackward0>)
    for(int bs=0; bs<5; bs++) {
        cout << " " << out_x[bs*n_vocab] << " " << out_x[bs*n_vocab+1] << " " << out_x[(bs+1) * n_vocab - 1] << endl;
    }
    closeModel(decoder);
}
