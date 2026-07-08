// TFLite-Micro wrapper (C++ TU behind a C API).
#pragma once
#include "model_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

int model_runner_init(void);  // 0 on success
// spec: N_MELS*N_FRAMES z-scored floats. logits_out: N_CLASSES dequantized floats.
// Quantization params are read from the model's own tensors (no drift possible).
int model_runner_invoke(const float *spec, float *logits_out);
int model_runner_arena_used(void);

#ifdef __cplusplus
}
#endif
