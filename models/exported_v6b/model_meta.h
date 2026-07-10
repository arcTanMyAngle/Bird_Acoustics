// Auto-generated model metadata — do not edit
// Source: bird_classifier_int8.tflite
#pragma once

// --- int8 quantization ---
#define MODEL_INPUT_SCALE   4.280050471e-02f
#define MODEL_INPUT_ZP      -19
#define MODEL_OUTPUT_SCALE  1.671903580e-01f
#define MODEL_OUTPUT_ZP     13

// --- decision rule: p = softmax(logits/T); detect iff
//     argmax != BACKGROUND_IDX && p[argmax] >= DETECT_TAU_PER_CLASS[argmax] ---
#define DETECT_TEMPERATURE  0.885382f
#define DETECT_TAU          0.600000f  // global fallback / legacy
#define BACKGROUND_IDX      1

// --- audio frontend contract (must match training exactly) ---
#define AUDIO_SAMPLE_RATE   16000
#define CLIP_SAMPLES        48000   // 3.0 s
#define N_FFT               512
#define HOP_LENGTH          256
#define N_MELS              64
#define N_FRAMES            188     // 1 + CLIP_SAMPLES/HOP (center=true)
#define N_FFT_BINS          257     // N_FFT/2 + 1
#define MEL_F_MAX           7000.0f  // realized in frontend_tables.h
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

// per-class softmax detection thresholds; index by argmax
static const float DETECT_TAU_PER_CLASS[N_CLASSES] = {
    0.600000f,  // american_crow
    0.950000f,  // background
    0.600000f,  // california_quail
    0.600000f,  // california_scrub_jay
    0.600000f,  // great_horned_owl
    0.600000f,  // killdeer
    0.600000f,  // mourning_dove
    0.600000f,  // red_tailed_hawk
    0.600000f,  // western_meadowlark
};
