# Next session — add an `other_bird` open-set class

Cold-start brief for the next working session. All plan phases 0–7 are green; v6b is
deployed and on-device parity PASSED (100% argmax, mean|logitΔ| 0.0282). Do NOT redo v6b.

## Goal
Add an `other_bird` open-set class so untrained birds are rejected instead of being
force-classified into one of the 8 species or background. This is the #1 real-world
robustness gap and a prerequisite before ANY fine-grained species work. Do NOT add named
species this session.

## Read first
`bird-acoustics-project-state` memory, then docs/PROGRESS.md, docs/EXPERIMENTS.md,
docs/MODEL_CONTRACT.md. The frontend contract is FROZEN
(16 kHz / 3.0 s / n_fft 512 / hop 256 / 64 mels / f_max 7000) — never change one side only.

## Two-clone / execution notes
- Live Python workspace = WSL clone:
  `wsl -d Ubuntu -- bash -lc 'cd ~/bird-detection && uv run python scripts/...'`
  (CPU ~17 s/epoch; dataset lives here under data/{raw,processed,cleaned,augmented}).
- Sync any edited script/artifact back to this Windows repo after; commit here.
- In compound `bash -lc '...'` calls shell variables get stripped — use Python heredocs.

## Plan of attack (investigate before committing to code)
1. **Scope the blast radius.** Class set goes 9→10, touching the whole contract:
   dataset_v3.py class list, train_v4.py, calibrate_rejection.py (per-class τ array +
   `background_idx`), export_v6.py, every generated firmware/main/model/*.h, and
   decision.c `DETECT_TAU_PER_CLASS[]`. Classes are SORTED — inserting `other_bird` shifts
   indices (background is currently idx 1). Audit every hardcoded class index first.
2. **Data.** Source held-out bird species NOT in the current 8 (check data/raw provenance
   and scripts/prepare_data.py). Balance the `other_bird` group count against existing
   classes; respect the grouping invariant (`{class}_{idx4}` prefix = one recording;
   `_wN`/`_augN` share the parent group; grouped stratified 70/15/15). Regenerate augmented
   from processed/cleaned — never train on mixed filename generations.
3. **Architecture of rejection — evaluate both, recommend one with a small experiment:**
   (a) explicit 10th softmax class trained on other_bird samples (simpler through the
   frozen export path), or (b) keep 9 classes + an open-set score (energy / max-logit /
   entropy threshold) calibrated on held-out species (may generalize better to truly-unseen
   birds). Justify, don't assert.
4. **Train + calibrate + digest (models/v7).** `scripts/summarize_run.py --append --run v7`.
   GATE: v6b's core numbers must NOT regress — owl_R ≥ 0.94, jay_P ≥ 0.90, test bg_FP@τ
   ≤ 2.2% @ recall ≥ 86% — AND held-out untrained birds must be rejected (routed to
   other_bird/background, not a confident species). If core metrics regress, STOP and
   report — don't ship. Report honestly; don't overclaim.

## If v7 cleanly adds rejection without regressing v6b — deploy
- `export_v6` → copy the 4 headers into firmware/main/model/ → rebuild:
  `cmd /c "call C:\Espressif\tools\Microsoft.v6.0.1_profile.bat && cd /d c:\Users\bornt\Desktop\Bird_Acoustics\firmware && idf.py build"`
- Check `model_runner_arena_used()` in the boot log; right-size `kArenaSize` if the
  class-count change moved it.
- Re-run Phase-6 on-device parity via the SD-swap flow. **`verify_device_parity.py`'s
  `--model` DEFAULTS to exported_v5 — always override** to `models/exported_v7/...int8.tflite`.
- Sync models/{v7,exported_v7} to the Windows repo and commit.

## Deferred (do NOT start this session)
Depthwise-separable backbone — its latency gate is already cleared (on-device pipeline
measured 440 ms/window, arena 262 KB in PSRAM), so it is an optional headroom optimization,
not a fix. Tackle only after `other_bird` is locked.

## Hardware note
The board is deployed with v6b and runs live on power-up (no card needed for inference;
SD only logs detections.csv + enables eval mode). Reflash only after new model headers.
