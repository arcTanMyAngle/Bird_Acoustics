#include "sd_logger.h"
#include "model_meta.h"

#include <stdio.h>
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
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

static const char *TAG = "sdlog";
static bool mounted;
static QueueHandle_t q;

typedef struct { int64_t t_ms; int class_idx; float prob; } row_t;

static void logger_task(void *arg)
{
    row_t r;
    int since_sync = 0;
    for (;;) {
        if (!xQueueReceive(q, &r, portMAX_DELAY)) continue;
        FILE *f = fopen(LOG_PATH, "a");
        if (!f) { ESP_LOGE(TAG, "open failed"); continue; }
        fprintf(f, "%lld,%s,%.3f\n", (long long)r.t_ms, CLASS_NAMES[r.class_idx], r.prob);
        fclose(f);
        if (++since_sync >= 10) since_sync = 0;   // fclose already flushes to card
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
    q = xQueueCreate(16, sizeof(row_t));
    xTaskCreatePinnedToCore(logger_task, "sd_log", 4096, NULL, 3, NULL, 0);
    ESP_LOGI(TAG, "SD mounted, logging to " LOG_PATH);
    return 0;
}

bool sd_logger_available(void) { return mounted; }

void sd_logger_log(int64_t t_ms, int class_idx, float prob)
{
    if (!mounted) return;
    row_t r = { t_ms, class_idx, prob };
    xQueueSend(q, &r, 0);   // drop rather than stall inference if the queue is full
}
