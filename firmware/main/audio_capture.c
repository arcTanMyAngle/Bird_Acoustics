#include "audio_capture.h"
#include "model_meta.h"

#include <string.h>
#include "driver/i2s_pdm.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define PDM_CLK_GPIO  GPIO_NUM_42
#define PDM_DATA_GPIO GPIO_NUM_41
#define RING_SAMPLES  (4 * AUDIO_SAMPLE_RATE)   // 4 s, 128 KB in PSRAM
#define CHUNK_SAMPLES 1024

static const char *TAG = "audio";
static int16_t *ring;
static volatile uint32_t wr_total;   // monotonic samples written
static i2s_chan_handle_t rx;

static void capture_task(void *arg)
{
    int16_t chunk[CHUNK_SAMPLES];
    size_t got;
    for (;;) {
        if (i2s_channel_read(rx, chunk, sizeof(chunk), &got, portMAX_DELAY) != ESP_OK)
            continue;
        uint32_t n = got / sizeof(int16_t);
        uint32_t pos = wr_total % RING_SAMPLES;
        uint32_t first = n < RING_SAMPLES - pos ? n : RING_SAMPLES - pos;
        memcpy(ring + pos, chunk, first * sizeof(int16_t));
        memcpy(ring, chunk + first, (n - first) * sizeof(int16_t));
        wr_total += n;   // single writer; readers only need a consistent tail
    }
}

int audio_capture_start(void)
{
    ring = heap_caps_calloc(RING_SAMPLES, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!ring) return -1;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx));

    i2s_pdm_rx_config_t pdm_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                   I2S_SLOT_MODE_MONO),
        .gpio_cfg = { .clk = PDM_CLK_GPIO, .din = PDM_DATA_GPIO },
    };
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx, &pdm_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));

    xTaskCreatePinnedToCore(capture_task, "audio_cap", 4096, NULL, 6, NULL, 0);
    ESP_LOGI(TAG, "PDM capture running (%d Hz)", AUDIO_SAMPLE_RATE);
    return 0;
}

int audio_capture_snapshot(int16_t *dst)
{
    uint32_t end = wr_total;
    if (end < CLIP_SAMPLES) return -1;
    uint32_t start = (end - CLIP_SAMPLES) % RING_SAMPLES;
    uint32_t first = CLIP_SAMPLES < RING_SAMPLES - start ? CLIP_SAMPLES
                                                         : RING_SAMPLES - start;
    memcpy(dst, ring + start, first * sizeof(int16_t));
    memcpy(dst + first, ring, (CLIP_SAMPLES - first) * sizeof(int16_t));
    return 0;
}
