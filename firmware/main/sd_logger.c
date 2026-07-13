#include "sd_logger.h"
#include "model_meta.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#define SD_CS_GPIO   21
#define SD_SCK_GPIO  7
#define SD_MISO_GPIO 8
#define SD_MOSI_GPIO 9
#define LOG_PATH     "/sdcard/detections.csv"
#define CLIP_DIR     "/sdcard/clips"

static const char *TAG = "sdlog";
static bool mounted;
static QueueHandle_t q;

// clip (if non-NULL) is a heap_caps buffer of clip_n int16 samples, owned by the
// logger task, which frees it after writing.
typedef struct {
    int64_t t_ms;
    int class_idx;
    float prob;
    int16_t *clip;
    uint32_t clip_n;
} row_t;

// Write int16 mono PCM as a canonical 16 kHz WAV. Returns 0 on success.
static int write_wav(const char *path, const int16_t *pcm, uint32_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint32_t data_bytes = n * 2;
    const uint32_t rate = AUDIO_SAMPLE_RATE;
    const uint32_t byte_rate = rate * 2;   // mono, 16-bit
    uint8_t h[44];
    memcpy(h, "RIFF", 4);
    uint32_t riff = 36 + data_bytes;
    h[4] = riff & 0xFF; h[5] = (riff >> 8) & 0xFF; h[6] = (riff >> 16) & 0xFF; h[7] = (riff >> 24) & 0xFF;
    memcpy(h + 8, "WAVE", 4);
    memcpy(h + 12, "fmt ", 4);
    h[16] = 16; h[17] = 0; h[18] = 0; h[19] = 0;   // fmt chunk size
    h[20] = 1;  h[21] = 0;                          // PCM
    h[22] = 1;  h[23] = 0;                          // mono
    h[24] = rate & 0xFF; h[25] = (rate >> 8) & 0xFF; h[26] = (rate >> 16) & 0xFF; h[27] = (rate >> 24) & 0xFF;
    h[28] = byte_rate & 0xFF; h[29] = (byte_rate >> 8) & 0xFF; h[30] = (byte_rate >> 16) & 0xFF; h[31] = (byte_rate >> 24) & 0xFF;
    h[32] = 2; h[33] = 0;                           // block align
    h[34] = 16; h[35] = 0;                          // bits per sample
    memcpy(h + 36, "data", 4);
    h[40] = data_bytes & 0xFF; h[41] = (data_bytes >> 8) & 0xFF; h[42] = (data_bytes >> 16) & 0xFF; h[43] = (data_bytes >> 24) & 0xFF;
    bool ok = fwrite(h, 1, 44, f) == 44 && fwrite(pcm, 1, data_bytes, f) == data_bytes;
    fclose(f);
    return ok ? 0 : -1;
}

static void logger_task(void *arg)
{
    row_t r;
    for (;;) {
        if (!xQueueReceive(q, &r, portMAX_DELAY)) continue;
        FILE *f = fopen(LOG_PATH, "a");
        if (f) {
            fprintf(f, "%lld,%s,%.3f\n", (long long)r.t_ms, CLASS_NAMES[r.class_idx], r.prob);
            fclose(f);   // fclose flushes to card
        } else {
            ESP_LOGE(TAG, "open failed");
        }
        if (r.clip) {
            char path[64];
            snprintf(path, sizeof(path), CLIP_DIR "/%lld_%s.wav",
                     (long long)r.t_ms, CLASS_NAMES[r.class_idx]);
            if (write_wav(path, r.clip, r.clip_n) == 0)
                ESP_LOGI(TAG, "saved %s", path);
            else
                ESP_LOGW(TAG, "clip write failed: %s", path);
            free(r.clip);
        }
    }
}

int sd_logger_init(void)
{
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = SPI2_HOST;

    spi_bus_config_t bus = {
        .mosi_io_num = SD_MOSI_GPIO,
        .miso_io_num = SD_MISO_GPIO,
        .sclk_io_num = SD_SCK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096,
    };
    esp_err_t err = spi_bus_initialize(host.slot, &bus, SDSPI_DEFAULT_DMA);
    if (err != ESP_OK) { ESP_LOGE(TAG, "spi init: %s", esp_err_to_name(err)); return -1; }

    sdspi_device_config_t slot = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot.gpio_cs = SD_CS_GPIO;
    slot.host_id = host.slot;

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 4,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_card_t *card;
    err = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot, &mount_cfg, &card);
    if (err != ESP_OK) { ESP_LOGW(TAG, "mount failed: %s", esp_err_to_name(err)); return -1; }

    mounted = true;
    mkdir(CLIP_DIR, 0777);   // harmless if it already exists
    q = xQueueCreate(16, sizeof(row_t));
    xTaskCreatePinnedToCore(logger_task, "sd_log", 4096, NULL, 3, NULL, 0);
    ESP_LOGI(TAG, "SD mounted, logging to " LOG_PATH ", clips to " CLIP_DIR);
    return 0;
}

bool sd_logger_available(void) { return mounted; }

void sd_logger_log(int64_t t_ms, int class_idx, float prob)
{
    if (!mounted) return;
    row_t r = { t_ms, class_idx, prob, NULL, 0 };
    xQueueSend(q, &r, 0);   // drop rather than stall inference if the queue is full
}

void sd_logger_log_clip(int64_t t_ms, int class_idx, float prob,
                        const int16_t *pcm, uint32_t n_samples)
{
    if (!mounted) return;
    // Copy the PCM so the inference task can reuse its buffer; the logger frees it.
    int16_t *copy = heap_caps_malloc(n_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!copy) { sd_logger_log(t_ms, class_idx, prob); return; }  // clip-less fallback
    memcpy(copy, pcm, n_samples * sizeof(int16_t));
    row_t r = { t_ms, class_idx, prob, copy, n_samples };
    if (xQueueSend(q, &r, 0) != pdTRUE) free(copy);   // drop rather than stall inference
}
