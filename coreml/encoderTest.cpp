#include "encoderWrapper.h"
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
    loadModel("./small", n_layer, n_state);

    float* melSegment = getOnes(1*80*3000);
    float* out_x = getOnes(1*1500*n_state);

    for(int i=0; i<5; i++) {
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        predictWith(melSegment, out_x);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Encoder " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    }

    // it should match pytorch output:
    // tensor([ 0.9057, -1.3382], grad_fn=<SliceBackward0>) tensor(7.6636, grad_fn=<SelectBackward0>)
    cout << " " << out_x[0] << " " << out_x[1];// << " " << out_x[bs * 51865 - 1];
    closeModel();
}
