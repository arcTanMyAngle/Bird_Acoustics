#include "eval_mode.h"
#include "frontend.h"
#include "model_meta.h"
#include "model_runner.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "esp_heap_caps.h"
#include "esp_log.h"

#define EVAL_DIR "/sdcard/eval"
#define OUT_PATH "/sdcard/eval_out.csv"
#define HOP_SAMPLES AUDIO_SAMPLE_RATE   // 1 s window hop

static const char *TAG = "eval";

// Minimal RIFF reader: 16-bit mono PCM at AUDIO_SAMPLE_RATE, chunk-walked.
static int16_t *wav_read(const char *path, uint32_t *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fclose(f); return NULL;
    }
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    int16_t *data = NULL;
    uint32_t n = 0;
    uint8_t ck[8];
    while (fread(ck, 1, 8, f) == 8) {
        uint32_t size = ck[4] | ck[5] << 8 | ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) {
            uint8_t fmt[16];
            if (size < 16 || fread(fmt, 1, 16, f) != 16) break;
            channels = fmt[2] | fmt[3] << 8;
            rate = fmt[4] | fmt[5] << 8 | fmt[6] << 16 | (uint32_t)fmt[7] << 24;
            bits = fmt[14] | fmt[15] << 8;
            fseek(f, size - 16, SEEK_CUR);
        } else if (!memcmp(ck, "data", 4)) {
            if (channels != 1 || rate != AUDIO_SAMPLE_RATE || bits != 16) break;
            data = heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
            if (data && fread(data, 1, size, f) == size)
                n = size / 2;
            else { free(data); data = NULL; }
            break;
        } else {
            fseek(f, size + (size & 1), SEEK_CUR);
        }
    }
    fclose(f);
    if (!data) ESP_LOGW(TAG, "%s: need 16-bit mono %d Hz PCM", path, AUDIO_SAMPLE_RATE);
    *n_out = n;
    return data;
}

bool eval_mode_requested(void)
{
    struct stat st;
    return stat(EVAL_DIR, &st) == 0 && S_ISDIR(st.st_mode);
}

int eval_mode_run(void)
{
    FILE *out = fopen(OUT_PATH, "w");
    if (!out) { ESP_LOGE(TAG, "cannot open " OUT_PATH); return 0; }
    fprintf(out, "file,window");
    for (int c = 0; c < N_CLASSES; c++) fprintf(out, ",logit_%s", CLASS_NAMES[c]);
    fprintf(out, ",pred\n");

    float *audio = heap_caps_malloc(CLIP_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);
    float *spec = heap_caps_malloc(N_MELS * N_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    float logits[N_CLASSES];

    int files = 0;
    DIR *dir = opendir(EVAL_DIR);
    struct dirent *e;
    while (dir && (e = readdir(dir))) {
        if (!strstr(e->d_name, ".wav") && !strstr(e->d_name, ".WAV")) continue;
        char path[300];
        snprintf(path, sizeof(path), EVAL_DIR "/%s", e->d_name);
        uint32_t n;
        int16_t *pcm = wav_read(path, &n);
        if (!pcm) continue;

        int windows = n >= CLIP_SAMPLES ? 1 + (n - CLIP_SAMPLES) / HOP_SAMPLES : 0;
        for (int w = 0; w < windows; w++) {
            const int16_t *seg = pcm + (long)w * HOP_SAMPLES;
            for (int i = 0; i < CLIP_SAMPLES; i++)
                audio[i] = seg[i] / 32768.0f;
            frontend_compute(audio, spec);
            if (model_runner_invoke(spec, logits) != 0) continue;
            int best = 0;
            for (int c = 1; c < N_CLASSES; c++)
                if (logits[c] > logits[best]) best = c;
            fprintf(out, "%s,%d", e->d_name, w);
            for (int c = 0; c < N_CLASSES; c++) fprintf(out, ",%.6f", logits[c]);
            fprintf(out, ",%s\n", CLASS_NAMES[best]);
        }
        free(pcm);
        files++;
        ESP_LOGI(TAG, "%s: %d windows", e->d_name, windows);
    }
    if (dir) closedir(dir);
    fclose(out);
    free(audio);
    free(spec);
    ESP_LOGI(TAG, "eval complete: %d files -> " OUT_PATH, files);
    return files;
}
