# Bird Acoustic Detection System

Real-time bird species classification on edge hardware using TinyML — a Seeed
XIAO ESP32-S3 Sense listens for bird calls, classifies them on-device with a
quantized CNN, and logs confirmed detections to a microSD card.

## Overview

The system captures audio via the XIAO's built-in PDM microphone, extracts a
log-mel spectrogram on-device, runs int8 inference with TFLite-Micro, and
applies a calibrated confidence threshold plus temporal smoothing before
logging a detection. Training runs offline in PyTorch and exports to a
75 KB int8 TFLite-Micro model; firmware is native ESP-IDF (no Arduino).

**Key metrics (v5, current):**
- Held-out test accuracy: **90.2%** across 9 classes (8 species + background), on a leakage-free grouped train/val/test split
- Background false-positive rate: **2.2%** per window at 84.7% bird recall (calibrated softmax threshold + 2-of-3 window smoothing)
- Model: 75 KB int8, ~65K params; full pipeline (frontend + inference) verified to run well under the old baseline
- Inference latency: **~12–15 s → sub-second** moving from Arduino (`TensorFlowLite_ESP32`, no hardware acceleration) to ESP-IDF + ESP-NN
- Power: ~150 mA active · Hardware cost: ~$50

Full gate-by-gate numbers and the model's exact contract (quantization
params, frontend math, decision rule) live in [`docs/PROGRESS.md`](docs/PROGRESS.md)
and [`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md). Toolchain setup and
day-to-day commands are in [`CLAUDE.md`](CLAUDE.md).

## Target Species

| Species | Scientific Name |
|---------|-----------------|
| American Crow | *Corvus brachyrhynchos* |
| California Quail | *Callipepla californica* |
| California Scrub-Jay | *Aphelocoma californica* |
| Great Horned Owl | *Bubo virginianus* |
| Killdeer | *Charadrius vociferus* |
| Mourning Dove | *Zenaida macroura* |
| Red-tailed Hawk | *Buteo jamaicensis* |
| Western Meadowlark | *Sturnella neglecta* |
| Background/Noise | — (rejection class) |

## Hardware

| Component | Purpose |
|-----------|---------|
| Seeed XIAO ESP32S3 Sense | MCU with built-in PDM mic, 8MB flash, 8MB octal PSRAM |
| 32GB microSD card (FAT32) | Detection logging |
| USB-C cable | Power + flashing |

## Project Structure

```
Bird_Acoustics/
├── pyproject.toml, uv.lock     # Python training deps (uv-managed)
├── Dockerfile                  # Reproducible training container
├── scripts/                    # Data pipeline, training, export, verification (see below)
├── models/
│   ├── v5/                     # Latest PyTorch checkpoint + training artifacts
│   └── exported_v5/            # Deployed int8 TFLite model + generated C headers
├── firmware/                   # ESP-IDF project (target: esp32s3)
│   ├── main/                   # Firmware source (frontend, TFLM runner, decision, SD logging)
│   │   └── model/              # Generated headers (model_data.h, model_meta.h, frontend_tables.h)
│   └── host_test/              # Host-buildable frontend for parity testing against Python
├── docs/
│   ├── PROGRESS.md             # Phase-by-phase gate log with measured numbers
│   └── MODEL_CONTRACT.md       # Frontend/quantization/decision-rule spec (source of truth)
├── CLAUDE.md                   # Toolchain, two-clone layout, build commands
└── data/                       # Audio datasets (not committed; see Data below)
```

## Data Pipeline

The full pipeline (data lives outside this repo — see **Data** below):

```
raw/ (Xeno-canto + ESC-50)
  → preprocess_audio.py    (16 kHz, 3 s clips, energy-gated windows, background gets 1 s-hop overlap)
  → augmented/             (augment_audio.py: noise/gain/pitch/stretch/filter, deterministic seed)
  → dataset_v3.py          (grouped stratified 70/15/15 split — no recording leaks across splits)
  → train_v4.py            (BirdClassifierCNN, ~65K params, weighted CE + label smoothing)
  → calibrate_rejection.py (temperature scaling + rejection threshold τ, fit on val, gated on test)
  → export_v6.py           (PT2E int8 quantization, MEAN-pooling ESP32 graph, header generation)
  → firmware/main/model/*.h
```

Each stage has a pass/fail gate — see [`docs/PROGRESS.md`](docs/PROGRESS.md) for
what was measured at each step (leakage check, honest test accuracy,
false-positive rate, quantization agreement, C-frontend-vs-PyTorch parity,
host end-to-end agreement).

### Quick start (training side)

```bash
uv sync
uv run python scripts/test_setup.py       # sanity-check the environment

# Full pipeline from raw audio (skip steps you've already run):
uv run python scripts/download_xenocanto.py
uv run python scripts/download_bg.py
uv run python scripts/preprocess_audio.py
uv run python scripts/augment_audio.py --seed 42 --mix-noise
uv run python scripts/create_manifest.py --data-dir data/augmented

uv run python scripts/verify_split_v3.py --data-dir data/augmented   # leakage gate
uv run python scripts/train_v4.py --data-dir data/augmented --output-dir models/v5
uv run python scripts/calibrate_rejection.py --model-dir models/v5  # FP-rate gate
uv run python scripts/export_v6.py --model-path models/v5/best_model.pth \
    --output-dir models/exported_v5
```

### Verification scripts (run before ever touching the board)

| Script | Gate |
|---|---|
| `verify_split_v3.py` | No recording group appears in more than one of train/val/test |
| `calibrate_rejection.py` | Background false-positive rate ≤ target on held-out test |
| `verify_frontend_parity.py` | C frontend (host-compiled) matches `AlignedMelSpectrogram` (torchaudio) |
| `verify_e2e_host.py` | C frontend → int8 TFLite matches the pure-Python reference pipeline |
| `verify_device_parity.py` | Device eval-mode output (from real hardware) matches the host reference |

## Firmware (ESP-IDF)

Native ESP-IDF (v6.0.1+), no Arduino. Pipeline: PDM mic → PSRAM ring buffer →
on-device log-mel frontend (reflect-pad, Hann window, real FFT, mel filterbank
and window exported directly from torchaudio for guaranteed parity) → int8
TFLite-Micro inference → calibrated softmax threshold + 2-of-3 window smoothing
→ async CSV logging to microSD. An eval mode replays WAVs staged on the SD
card through the identical pipeline for device-vs-host validation without
needing live acoustics.

```bash
# one-time: install ESP-IDF, see CLAUDE.md
idf.py set-target esp32s3
idf.py build
idf.py -p COMx flash monitor
```

Board pins (XIAO ESP32S3 Sense): PDM mic CLK=GPIO42 DATA=GPIO41; SD (SDSPI)
CS=GPIO21 SCK=GPIO7 MISO=GPIO8 MOSI=GPIO9.

## Model Architecture

Compact CNN (`BirdClassifierCNN`, ~65K params), 4 conv blocks (16→32→64→64
channels) with batch norm, ReLU, and max pooling, global mean pooling, then a
small FC head. Trained with weighted cross-entropy (class imbalance) plus
label smoothing; exported via PT2E static per-channel symmetric int8
quantization with real-spectrogram calibration.

### Audio Frontend

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 kHz |
| Clip duration | 3.0 s (48000 samples) |
| FFT size | 512 (257 bins) |
| Hop length | 256 (188 frames) |
| Mel bands | 40 |
| Log scale | `10·log10`, clamped to global max − 80 dB |
| Normalization | Per-sample z-score (unbiased std) |

The exact frontend math (including the reflect-padding and z-score details
that are easy to get subtly wrong on a re-implementation) is documented in
[`docs/MODEL_CONTRACT.md`](docs/MODEL_CONTRACT.md) and enforced by the parity
gates above.

## Data

Raw and processed audio are **not committed** to this repo (`data/` is
gitignored). The dataset is built from public sources:

- [Xeno-canto](https://xeno-canto.org/) — bird call recordings
- [ESC-50](https://github.com/karolpiczak/ESC-50) — environmental sounds (background/negative class)

Run `scripts/prepare_data.py` (or the individual download/preprocess/augment
scripts above) to rebuild the dataset locally.

## References

- [XIAO ESP32S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/) — hardware documentation
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/) — official framework docs
- [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) — TFLite-Micro port with ESP-NN acceleration

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

arcTanMyAngle

---

*An end-to-end TinyML project: data collection, leakage-free training,
calibrated rejection, int8 quantization, and native ESP-IDF edge deployment.*
