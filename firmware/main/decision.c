#include "decision.h"
#include <math.h>

static int hist[SMOOTH_N] = { -1, -1, -1 };  // window-hit class per window, -1 = none
_Static_assert(SMOOTH_N == 3, "update hist initializer");
static int hist_pos;
static int last_emit[N_CLASSES];    // window counter at last confirmation
static int window_count;

decision_t decision_update(const float *logits)
{
    decision_t d = { .class_idx = 0, .prob = 0.0f, .window_hit = false, .confirmed = false };

    float m = logits[0];
    for (int c = 1; c < N_CLASSES; c++)
        if (logits[c] > m) m = logits[c];
    float sum = 0.0f, p[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) {
        p[c] = expf((logits[c] - m) / DETECT_TEMPERATURE);
        sum += p[c];
    }
    int best = 0;
    for (int c = 1; c < N_CLASSES; c++)
        if (p[c] > p[best]) best = c;

    d.class_idx = best;
    d.prob = p[best] / sum;
    // Per-class threshold (v6): low tau for high-precision classes (great_horned_owl) recovers
    // recall; high tau for over-predicted classes (california_scrub_jay) lifts precision.
    d.window_hit = (best != BACKGROUND_IDX) && (d.prob >= DETECT_TAU_PER_CLASS[best]);

    hist[hist_pos] = d.window_hit ? best : -1;
    hist_pos = (hist_pos + 1) % SMOOTH_N;
    window_count++;

    if (d.window_hit) {
        int votes = 0;
        for (int i = 0; i < SMOOTH_N; i++)
            if (hist[i] == best) votes++;
        if (votes >= SMOOTH_M && window_count - last_emit[best] > SMOOTH_N) {
            d.confirmed = true;
            last_emit[best] = window_count;
        }
    }
    return d;
}
