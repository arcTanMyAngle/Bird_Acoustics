// Bird acoustic detector — XIAO ESP32S3 Sense (ESP-IDF).
// Live mode: PDM mic -> log-mel -> int8 TFLM -> calibrated threshold + M-of-N
//            smoothing -> CSV log on microSD.
// Eval mode: if /sdcard/eval/ exists, replays WAVs through the identical
//            pipeline for host-vs-device parity checks, then idles.
#include <stdio.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "audio_capture.h"
#include "decision.h"
#include "eval_mode.h"
#include "frontend.h"
#include "model_meta.h"
#include "model_runner.h"
#include "sd_logger.h"

#define HOP_MS 1000   // window hop: run the 3 s window once per second

static const char *TAG = "bird";

static void inference_task(void *arg)
{
    int16_t *pcm = heap_caps_malloc(CLIP_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    float *audio = heap_caps_malloc(CLIP_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
    float *spec = heap_caps_malloc(N_MELS * N_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    float logits[N_CLASSES];
    int windows = 0;

    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(HOP_MS));
        if (audio_capture_snapshot(pcm) != 0) continue;   // ring still filling

        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < CLIP_SAMPLES; i++)
            audio[i] = pcm[i] / 32768.0f;
        frontend_compute(audio, spec);
        if (model_runner_invoke(spec, logits) != 0) continue;
        decision_t d = decision_update(logits);
        int64_t dt_ms = (esp_timer_get_time() - t0) / 1000;

        if (d.confirmed) {
            int64_t now_ms = esp_timer_get_time() / 1000;
            ESP_LOGI(TAG, "DETECTED %s p=%.2f (pipeline %lld ms)",
                     CLASS_NAMES[d.class_idx], d.prob, dt_ms);
            sd_logger_log(now_ms, d.class_idx, d.prob);
        }
        if (++windows % 60 == 0)
            ESP_LOGI(TAG, "alive: %d windows, last pipeline %lld ms, arena %d B",
                     windows, dt_ms, model_runner_arena_used());
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "bird detector starting");
    frontend_init();
    if (model_runner_init() != 0) {
        ESP_LOGE(TAG, "model init failed — halting");
        return;
    }

    bool sd_ok = sd_logger_init() == 0;
    if (sd_ok && eval_mode_requested()) {
        ESP_LOGI(TAG, "eval mode: processing /sdcard/eval");
        eval_mode_run();
        ESP_LOGI(TAG, "eval done — idle (remove /sdcard/eval and reboot for live mode)");
        return;
    }
    if (!sd_ok)
        ESP_LOGW(TAG, "no SD card — detections logged to console only");

    if (audio_capture_start() != 0) {
        ESP_LOGE(TAG, "mic init failed — halting");
        return;
    }
    xTaskCreatePinnedToCore(inference_task, "inference", 16384, NULL, 5, NULL, 1);
}
