// Log-mel frontend replicating training's AlignedMelSpectrogram exactly:
// reflect pad (center=true) -> periodic Hann -> |FFT|^2 -> mel (slaney/HTK,
// exported tables) -> 10*log10 clamped to global_max - TOP_DB -> per-sample
// z-score with unbiased (N-1) std.  All dims come from model_meta.h.
#pragma once
#include "model_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

void frontend_init(void);
// audio: CLIP_SAMPLES mono floats in [-1, 1].
// spec_out: N_MELS * N_FRAMES floats, mel-major (matches model input 1x1x40x188).
void frontend_compute(const float *audio, float *spec_out);

#ifdef __cplusplus
}
#endif
