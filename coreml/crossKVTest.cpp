#include "crossKV.h"
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

    // small model
    int n_layer = 12;
    int n_state = 768;
    int bs = 5;
    int text_offset = 10; // only for test
    const void* crossKV = loadModel("./small/CoremlCrossKV.mlmodelc", n_layer, n_state);

    float* xa = getOnes(1 * 1500 * n_state);

    float* out_cross_k_caches = getOnes( n_layer * 1500 * n_state);
    float* out_cross_v_caches = getOnes( n_layer * 1500 * n_state);

    for(int i=0; i<5; i++) {
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        predictWith(crossKV, // model
                xa,
                out_cross_k_caches, // outputs
                out_cross_v_caches // outputs
                );
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "crossKV " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << ", isNewCKV=" << (i==0) << endl;
    }

    for(int bs=0; bs<5; bs++) {
        cout << " " << out_cross_k_caches[0] << " " << out_cross_k_caches[1] << endl;
        cout << " " << out_cross_v_caches[0] << " " << out_cross_v_caches[1] << endl;
    }
    closeModel(crossKV);
}
