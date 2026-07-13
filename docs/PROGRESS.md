# Phase gate log

- 2026-07-12 P6 ✅ On-device parity (v6b, 64-mel, on-board SD eval): board replayed all 27 eval clips through the identical PDM→frontend→int8 TFLM pipeline (arena 262072/327680 B, in 0.042801/−19) → eval_out.csv. `verify_device_parity.py --model models/exported_v6b/...int8.tflite`: **argmax agreement 100.0% (27/27), mean |logit Δ| 0.0282 (0.2 int8 LSB)** — gate ≥99%/≤0.10 PASSED. Fixed script's hardcoded `reshape(1,1,40,188)` → `reshape(inp["shape"])` (was v5-only; broke on 64-mel). Live-mode detections.csv on-board also sane (crow 0.998, scrub_jay 0.72, meadowlark 0.86 @ τ=0.60).

- 2026-07-07 P5 ✅ `idf.py build` clean on ESP-IDF v6.0.1: bird_detector.bin 502 KB (84% partition free). esp-dsp dropped (xtensa-gcc 15.2 ICE on dspi_conv_f32_ansi.c — deterministic); device uses the same portable FFT that passed host parity. IDF v6 note: I2S needs `esp_driver_i2s` in PRIV_REQUIRES. Eval SD set staged at WSL ~/bird-detection/eval_sd/eval (27 clips). Next: P6 user flashes + runs eval mode.

- 2026-07-07 P4 ✅ Export v5 → models/exported_v5: int8 75 KB, PT↔TFLite agreement 99%, 0% accuracy drop; model_meta.h (in 0.05565/−9, out 0.05038/−22, T=0.651, τ=0.62) + frontend_tables.h generated and copied to firmware/main/model/. Host e2e gate ✅: C-frontend→int8 model = 100% argmax agreement (45 clips), 0 background false-fires.

- 2026-07-07 P2 ✅ v5 retrain (9-class, weighted CE + label_smoothing 0.1, sampler off): held-out TEST acc 90.2% (macro F1 0.89), val 91.5%, synthetic bg rejection 96% (FP 4%, was 14% in v4_fixed). Weakest class: california_scrub_jay (test precision 0.60). models/v5/.
- 2026-07-07 P3 ✅ Calibration: T=0.651, tau=0.62 (val-anchored min-FP + 0.10 margin). TEST: FP 2.2%, bird recall 84.7% (gate ≤5%). models/v5/calibration.json.

- 2026-07-07 P5(frontend) ✅ C frontend parity vs torchaudio: worst max|Δ| = 1.4e-04 over 27 real clips (tolerance 1e-2) — exact reflect-pad/Hann/FFT/mel/dB/z-score chain. Firmware written (~10 files); export smoke: 75 KB int8, 99% PT↔TFLite agreement, ops = CONV_2D, DW_CONV_2D, DEQUANT, FC, MAXPOOL, MUL, QUANT, RESHAPE, SUM, TRANSPOSE.

- 2026-07-07 P1 ✅ Data regenerated deterministically (seed 42, --mix-noise): 2667 processed (background 300 via 1s-hop overlap windows, was 100) → 8001 augmented. Bijection: 789 groups, 0 conflicts. Split gate: 0 group overlap train/val/test (5544/1206/1251), all 9 classes in all splits. Old contaminated set kept at data/augmented_old_mixed (delete after P2 gate).

- 2026-07-07 P0 ✅ Recon: WSL clone is live lineage (dataset_v3/train_v4/export_v6, uncommitted). Class set = 9 (8 birds + background). Grouping bijection verified: 2467 processed files → 789 groups, 0 conflicts. augmented/ contaminated with 2 filename generations → must regenerate. ESP-IDF not installed (Windows or WSL). v4_fixed prior result: 86% bg rejection (14% FP) on contaminated data.
