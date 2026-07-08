// microSD (SDSPI) mount + asynchronous CSV detection log.
#pragma once
#include <stdbool.h>
#include <stdint.h>

int sd_logger_init(void);          // mounts /sdcard; 0 on success
bool sd_logger_available(void);
// Queues one detection row (written by a low-priority task, off the inference path).
void sd_logger_log(int64_t t_ms, int class_idx, float prob);
