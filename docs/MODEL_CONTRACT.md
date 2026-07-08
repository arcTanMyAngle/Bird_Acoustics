# Model contract (v5, exported 2026-07-07)

Human-readable mirror of the generated headers in `firmware/main/model/` —
regenerate via `scripts/export_v6.py`, never hand-edit either side.

## Audio frontend (device must match training bit-for-bit)
| Param | Value |
|---|---|
| Sample rate | 16000 Hz mono |
| Window | 3.0 s (48000 samples), 1 s hop between inferences |
| STFT | n_fft 512, hop 256, center=true (reflect pad), periodic Hann |
| Mel | 40 bins, slaney norm, HTK scale, f_max 8000 (tables exported from torchaudio) |
| Log | 10·log10(max(power,1e-10)), clamped to global_max − 80 dB |
| Normalize | per-sample z-score, unbiased (N−1) std |
| Input tensor | (1,1,40,188) int8, scale 0.05565, zp −9 |

## Model
`BirdClassifierESP32Mean` (~65K params), int8 PTQ (per-channel symmetric,
200 real calibration samples), 75 KB flatbuffer, MEAN-op global pooling.
Output (1,9) int8 logits, scale 0.05038, zp −22.
Ops: CONV_2D, DEPTHWISE_CONV_2D, DEQUANTIZE, FULLY_CONNECTED, MAX_POOL_2D, MUL,
QUANTIZE, RESHAPE, SUM, TRANSPOSE.

## Classes (sorted; index = model output index)
0 american_crow · 1 **background** · 2 california_quail · 3 california_scrub_jay ·
4 great_horned_owl · 5 killdeer · 6 mourning_dove · 7 red_tailed_hawk ·
8 western_meadowlark

## Decision rule (calibrated on val, evaluated once on test)
`p = softmax(logits / 0.651)`; window hit iff `argmax != background` and
`p[argmax] ≥ 0.62`; confirmed detection = same class in ≥2 of last 3 windows
(+3-window per-class refractory).

**Measured (held-out test):** accuracy 90.2%, background false-positive rate
2.2%, bird recall 84.7% (per-window, before M-of-N smoothing).
PT↔TFLite agreement 99%; C-frontend parity max|Δ| 1.4e-04;
host end-to-end argmax agreement 100%.
