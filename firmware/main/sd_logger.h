// microSD (SDSPI) mount + asynchronous CSV detection log.
#pragma once
#include <stdbool.h>
#include <stdint.h>

int sd_logger_init(void);          // mounts /sdcard; 0 on success
bool sd_logger_available(void);
// Queues one detection row (written by a low-priority task, off the inference path).
void sd_logger_log(int64_t t_ms, int class_idx, float prob);
// Same, but also saves the triggering audio as /sdcard/clips/<t_ms>_<class>.wav
// (16-bit mono AUDIO_SAMPLE_RATE PCM). Copies `pcm` internally; caller keeps ownership.
void sd_logger_log_clip(int64_t t_ms, int class_idx, float prob,
                        const int16_t *pcm, uint32_t n_samples);
