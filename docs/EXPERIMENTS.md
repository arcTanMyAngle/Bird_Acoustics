# Experiment ledger

One scalar row per run. Full confusion matrices live on disk in `models/<run>/` — never paste the 9x9 grid into a session; run `scripts/summarize_run.py --model-dir models/<run>` for the ≤15-line digest, and `--append` to add the row below automatically.

`owl_R` = great_horned_owl recall (false-negative symptom); `jay_P` = california_scrub_jay precision (over-prediction / FP-sink symptom); `bg_FP@tau` = calibrated background false-alarm rate on the held-out test split; `5dB_det` = bird-detection accuracy at 5 dB SNR.

| run | date | frontend (nfft/mels/fmax) | arch | loss | data | val_mF1 | test_acc | owl_R | jay_P | bg_FP@tau | 5dB_det | ms | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v5 | 2026-07-07 | 512/40/8000 | std-cnn ~63K | cw-ce+ls | aug 5–20dB p0.3 | — | 90.25 | 0.829 | 0.604 | 2.2% | 75 | ~sub-s | baseline; global tau=0.62; scrub_jay FP-sink, owl FN, low-SNR weak |
| v5+pc | 2026-07-08 | 512/40/8000 | std-cnn ~63K | cw-ce+ls | aug 5–20dB p0.3 | — | 90.25 | 0.829 | 0.604 | **0.7%** | 75 | ~sub-s | v5 model + per-class tau (scrub_jay 0.92) → FP 2.2%→0.7% same recall; **shipped in firmware** |
| v6 | 2026-07-08 | **512/64/7000** | std-cnn 63K | **cb-focal** | **cleaned + 0–20dB p0.5, jay/owl 4×** | 0.903(test) | 90.53 | **0.877** | **0.899** | 4.2% | **95** | tbd | class much better (jay P .60→.90, owl R .83→.88, 5dB det 75→95); detection FP↑ (needs calib retune: ls→0, refit T) |
