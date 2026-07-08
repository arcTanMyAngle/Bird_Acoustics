// PDM mic capture -> PSRAM ring buffer (XIAO ESP32S3 Sense: CLK=42, DATA=41).
#pragma once
#include <stdint.h>

int audio_capture_start(void);
// Copies the most recent CLIP_SAMPLES into dst. Returns 0, or -1 if the ring
// hasn't filled 3 s yet.
int audio_capture_snapshot(int16_t *dst);
