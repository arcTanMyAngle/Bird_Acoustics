// Device-parity eval: runs the identical frontend+model pipeline over WAVs in
// /sdcard/eval/ and writes per-window logits to /sdcard/eval_out.csv, so device
// results can be diffed against the host ai-edge-litert reference bit-for-bit
// (within int8 tolerance) with no acoustics involved.
#pragma once
#include <stdbool.h>

bool eval_mode_requested(void);   // true if /sdcard/eval exists
int eval_mode_run(void);          // returns number of WAVs processed
