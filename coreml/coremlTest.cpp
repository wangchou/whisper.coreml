#include "coreml.h"
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

enum ModelSize {Tiny, Base, Small, Medium, Large, Turbo};
int n_audio_layers[] = {4, 6, 12, 24, 32, 32};
int n_text_layers[] = {4, 6, 12, 24, 32, 4};
int n_states[] = {384, 512, 768, 1024, 1280, 1280};
int n_heads[] = {6, 8, 12, 16, 20, 20};
int n_alignment_heads[] = {6, 8, 10, 6, 23, 23};
int n_melss[] = {80, 80, 80, 80, 80, 120};


// This only test the prediction speed,
// make sure all runs on ANE correctly
int main() {
    enum ModelSize modelSize = Tiny;
    int n_audio_layer = n_audio_layers[modelSize];
    int n_text_layer = n_text_layers[modelSize];
    int n_state = n_states[modelSize];
    int n_head = n_heads[modelSize];
    int n_alignment_head = n_alignment_heads[modelSize];
    int bs = 1;
    int beam_idx = 0; // decoder256Predict
    int text_offset = 10;
    int n_vocab = 51865; //multi-lang: 51865, en only: 51864
    int max_n_ctx = 256;
    int n_mels = n_melss[modelSize]; // turbo: 128, others: 80

    for(int run=0; run<2; run++) {
        cout << "/////////" << endl;
        cout << "/// Run " << run << endl;
        cout << "/////////" << endl;
        // Encoder
        loadEncoder("./tiny", n_audio_layer, n_state, n_mels);

        float* melSegment = getOnes(1*n_mels*3000);
        for(int i=0; i<3; i++) {
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            encoderPredict(melSegment);
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "Encoder " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
        }

        // crossKV
        cout << "---" << endl;
        loadCrossKV("./tiny/CrossKV.mlmodelc", n_text_layer, n_state);
        float* out_cross_k_caches = getOnes( n_text_layer * 1500 * n_state);
        float* out_cross_v_caches = getOnes( n_text_layer * 1500 * n_state);
        for(int i=0; i<3; i++) {
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            crossKVPredict();
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "crossKV " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
        }

        //decoder256
        cout << "---" << endl;
        loadDecoder256("./tiny/Decoder256.mlmodelc", n_text_layer, n_state, n_head, n_alignment_head, bs);

        float* x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
        float* qk_mask = getOnes(max_n_ctx * max_n_ctx); // (256, 256)

        float* out_x = getOnes(bs * max_n_ctx * n_state); // (bs, 1, n_state)
        float* out_cross_head_weights = getOnes(n_alignment_head * max_n_ctx * 1500);
        for(int i=0; i<3; i++) {
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            decoder256Predict(x, qk_mask, out_x, out_cross_head_weights, beam_idx);
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "Decoder256 " << chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
        }

        //decoder1
        cout << "---" << endl;
        loadDecoder1("./tiny/Decoder.mlmodelc", n_text_layer, n_state, n_head, n_vocab);

        float* qk_mask1 = getOnes(450); // (1, 449)
        float* out_x1 = getOnes(bs * n_vocab); // (bs, 1, n_state)

        for(int i=0; i<5; i++) {
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            decoder1Predict(x, qk_mask1, text_offset, out_x1);
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "decoder1 " << chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << endl;
        }
        cout << "" << endl;
    }

    closeEncoder();
    closeCrossKV();
    closeDecoder256();
    closeDecoder1();
}
