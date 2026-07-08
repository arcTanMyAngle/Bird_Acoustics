// Host parity harness: raw float32 audio in -> z-scored log-mel float32 out.
// Usage: parity_main <in.f32 (48000 floats)> <out.f32 (40*188 floats)>
#include <stdio.h>
#include <stdlib.h>
#include "frontend.h"

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s in.f32 out.f32\n", argv[0]); return 2; }

    static float audio[CLIP_SAMPLES], spec[N_MELS * N_FRAMES];
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(audio, sizeof(float), CLIP_SAMPLES, f) != CLIP_SAMPLES) {
        fprintf(stderr, "failed to read %d floats from %s\n", CLIP_SAMPLES, argv[1]);
        return 1;
    }
    fclose(f);

    frontend_init();
    frontend_compute(audio, spec);

    f = fopen(argv[2], "wb");
    fwrite(spec, sizeof(float), N_MELS * N_FRAMES, f);
    fclose(f);
    return 0;
}
