// Portable radix-2 complex FFT — used on BOTH host and device (esp-dsp was
// dropped: gcc 15.2 ICE; see idf_component.yml). ~30 µs/frame at 240 MHz.
#include "frontend_fft.h"
#include <math.h>

#define N 512
static float tw_re[N / 2], tw_im[N / 2];

void rfft512_init(void)
{
    for (int k = 0; k < N / 2; k++) {
        tw_re[k] = cosf(-2.0f * (float)M_PI * k / N);
        tw_im[k] = sinf(-2.0f * (float)M_PI * k / N);
    }
}

void rfft512_power(const float *in, float *out)
{
    static float re[N], im[N];
    // bit-reversed copy, imag = 0
    for (int i = 0; i < N; i++) {
        unsigned r = 0, x = i;
        for (int b = 0; b < 9; b++) { r = (r << 1) | (x & 1); x >>= 1; }
        re[r] = in[i];
        im[r] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        int step = N / len;
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < len / 2; j++) {
                float wr = tw_re[j * step], wi = tw_im[j * step];
                int a = i + j, b = a + len / 2;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
    for (int k = 0; k <= N / 2; k++)
        out[k] = re[k] * re[k] + im[k] * im[k];
}
