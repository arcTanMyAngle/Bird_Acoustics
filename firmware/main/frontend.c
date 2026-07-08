#include "frontend.h"
#include "frontend_fft.h"
#include "frontend_tables.h"
#include <math.h>
#include <string.h>

#define PAD (N_FFT / 2)
#define PADDED_LEN (CLIP_SAMPLES + 2 * PAD)

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
static float *padded;   // 48512 floats (~190 KB) -> PSRAM
#else
static float padded_buf[PADDED_LEN];
static float *padded = padded_buf;
#endif

void frontend_init(void)
{
#ifdef ESP_PLATFORM
    padded = heap_caps_malloc(PADDED_LEN * sizeof(float), MALLOC_CAP_SPIRAM);
#endif
    rfft512_init();
}

void frontend_compute(const float *audio, float *spec_out)
{
    // torch reflect padding: left = audio[PAD..1], right = audio[n-2..n-1-PAD]
    for (int i = 0; i < PAD; i++) {
        padded[i] = audio[PAD - i];
        padded[PAD + CLIP_SAMPLES + i] = audio[CLIP_SAMPLES - 2 - i];
    }
    memcpy(padded + PAD, audio, CLIP_SAMPLES * sizeof(float));

    float frame[N_FFT], power[N_FFT_BINS];
    for (int t = 0; t < N_FRAMES; t++) {
        const float *src = padded + (long)t * HOP_LENGTH;
        for (int i = 0; i < N_FFT; i++)
            frame[i] = src[i] * HANN_WINDOW[i];
        rfft512_power(frame, power);

        for (int m = 0; m < N_MELS; m++) {
            const float *w = MEL_FB_WEIGHTS + MEL_FB_OFFSET[m];
            const float *p = power + MEL_FB_START[m];
            float acc = 0.0f;
            for (int k = 0; k < MEL_FB_LEN[m]; k++)
                acc += w[k] * p[k];
            spec_out[m * N_FRAMES + t] = acc;
        }
    }

    // AmplitudeToDB(power): 10*log10(max(x, 1e-10)), clamped to global max - TOP_DB
    const int n = N_MELS * N_FRAMES;
    float gmax = -INFINITY;
    for (int i = 0; i < n; i++) {
        float v = 10.0f * log10f(fmaxf(spec_out[i], 1e-10f));
        spec_out[i] = v;
        if (v > gmax) gmax = v;
    }
    const float floor_db = gmax - TOP_DB;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (spec_out[i] < floor_db) spec_out[i] = floor_db;
        sum += spec_out[i];
    }

    // per-sample z-score, unbiased std (torch .std() default)
    const float mean = (float)(sum / n);
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = spec_out[i] - mean;
        var += d * d;
    }
    const float std = sqrtf((float)(var / (n - 1)));
    const float inv = 1.0f / (std + 1e-8f);
    for (int i = 0; i < n; i++)
        spec_out[i] = (spec_out[i] - mean) * inv;
}
