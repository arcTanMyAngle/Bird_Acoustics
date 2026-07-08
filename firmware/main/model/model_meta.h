// Auto-generated model metadata — do not edit
// Source: bird_classifier_int8.tflite
#pragma once

// --- int8 quantization ---
#define MODEL_INPUT_SCALE   5.565034598e-02f
#define MODEL_INPUT_ZP      -9
#define MODEL_OUTPUT_SCALE  5.037713796e-02f
#define MODEL_OUTPUT_ZP     -22

// --- decision rule: p = softmax(logits/T); detect iff
//     argmax != BACKGROUND_IDX && p[argmax] >= DETECT_TAU ---
#define DETECT_TEMPERATURE  0.651491f
#define DETECT_TAU          0.620000f
#define BACKGROUND_IDX      1

// --- audio frontend contract (must match training exactly) ---
#define AUDIO_SAMPLE_RATE   16000
#define CLIP_SAMPLES        48000   // 3.0 s
#define N_FFT               512
#define HOP_LENGTH          256
#define N_MELS              40
#define N_FRAMES            188     // 1 + CLIP_SAMPLES/HOP (center=true)
#define N_FFT_BINS          257     // N_FFT/2 + 1
#define TOP_DB              80.0f   // dB clamp below global max

#define N_CLASSES           9
static const char *const CLASS_NAMES[N_CLASSES] = {
    "american_crow",
    "background",
    "california_quail",
    "california_scrub_jay",
    "great_horned_owl",
    "killdeer",
    "mourning_dove",
    "red_tailed_hawk",
    "western_meadowlark",
};
