# Bird Acoustics — TinyML bird-call detector (XIAO ESP32-S3 Sense)

## Two-clone layout (critical)
- **This Windows clone** (`c:\Users\bornt\Desktop\Bird_Acoustics`): the canonical, cleaned-up repo — firmware, docs, and the current golden-path scripts/models are all synced here. ESP-IDF toolchain lives on Windows.
- **WSL clone** (`\\wsl.localhost\Ubuntu\home\arctan\bird-detection`, same GitHub repo): the **live** Python workspace where training/export actually execute — has its own uncommitted history (older script versions, one-off experiment dirs under `models/`). Dataset lives only there (`data/{raw,processed,augmented}`, ~4.7 GB). Run: `wsl -d Ubuntu -- bash -c 'cd ~/bird-detection && uv run python scripts/...'`
- From Git Bash, prefix WSL commands with `MSYS_NO_PATHCONV=1` when passing Linux paths.
- After any WSL-side script edit or retrain, re-sync the changed file(s)/artifacts into this Windows repo — this repo is what's under version control.

## Current pipeline lineage
`prepare_data.py` → `data/processed` → `augment_audio.py --seed 42 --mix-noise` → `data/augmented` → `dataset_v3.py` (grouped stratified 70/15/15 train/val/noise_test) → `train_v4.py` → `export_v6.py` (**MEAN-op global pooling — AvgPool2d(5,23) took 11.8 s on ESP32; never use it**) → int8 TFLite (~75 KB) → C headers → `firmware/main/model/`.

## The model contract (frontend params — NEVER change one side only)
16 kHz mono · 3.0 s window (48000) · n_fft 512 · hop 256 · n_mels 40 · power 2 · center=True (reflect pad) · norm='slaney' · mel_scale='htk' · f_max 8000 · AmplitudeToDB top_db=80 (clamp vs global max) · per-sample z-score (torch unbiased std) → input (1,1,40,188) int8. Defined in `AlignedMelSpectrogram` (dataset_v3.py). Mirrored in generated `firmware/main/model/*.h` — regenerate via export, never hand-edit.

## Classes (9 = 8 species + background, sorted)
american_crow, background, california_quail, california_scrub_jay, great_horned_owl, killdeer, mourning_dove, red_tailed_hawk, western_meadowlark

## Grouping/leakage invariant
Split is grouped by `{class}_{idx4}` filename prefix = one source recording (bijection verified on processed/, 789 groups). `_wN` windows and `_augN` augmentations must share their parent's group. **Never train on `data/augmented` containing mixed filename generations** — wipe & regenerate from processed/ if in doubt.

## Board facts (XIAO ESP32S3 Sense)
PDM mic CLK=GPIO42 DATA=GPIO41 · SD (SDSPI) CS=GPIO21 SCK=GPIO7 MISO=GPIO8 MOSI=GPIO9 · 8 MB flash · 8 MB octal PSRAM · target `esp32s3`.

## ESP-IDF toolchain (Windows)
ESP-IDF **v6.0.1** installed via EIM at `C:\esp\v6.0.1\esp-idf` (tools in `C:\Espressif\tools`). Build:
`cmd /c "call C:\Espressif\tools\Microsoft.v6.0.1_profile.bat && cd /d c:\Users\bornt\Desktop\Bird_Acoustics\firmware && idf.py build"`
Managed deps: esp-nn 1.2.3 (via esp-tflite-micro 1.3.7). **esp-dsp is intentionally NOT a dependency** — xtensa-gcc 15.2 (bundled with IDF v6.0.1) segfaults compiling `dspi_conv_f32_ansi.c`; the firmware uses the portable FFT in `frontend_fft_ref.c` instead (same one proven against torchaudio in the host parity test — `firmware/host_test/`). (A partial v5.5 clone also exists at C:\Espressif\frameworks\esp-idf — unused.)

## Docs
- `docs/PROGRESS.md` — one line per passed phase gate (measured numbers).
- `docs/MODEL_CONTRACT.md` — human-readable mirror of generated model_meta.h.
- Plan: `~/.claude/plans/i-am-developing-a-curried-quokka.md` (phases 0–7 with gates).
