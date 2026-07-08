// Calibrated decision rule + temporal smoothing.
// Per window: p = softmax(logits / DETECT_TEMPERATURE);
//   window hit  iff argmax != BACKGROUND_IDX && p[argmax] >= DETECT_TAU
//   confirmed   iff same class hits >= SMOOTH_M of last SMOOTH_N windows
//               (with a SMOOTH_N-window refractory per class)
#pragma once
#include <stdbool.h>
#include "model_meta.h"

#define SMOOTH_N 3
#define SMOOTH_M 2

typedef struct {
    int class_idx;      // argmax class of this window
    float prob;         // calibrated p[argmax]
    bool window_hit;    // passed tau + not background
    bool confirmed;     // M-of-N smoothing fired this window
} decision_t;

#ifdef __cplusplus
extern "C" {
#endif
decision_t decision_update(const float *logits);
#ifdef __cplusplus
}
#endif
