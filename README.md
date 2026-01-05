# California Bird Acoustic Detection System

Real-time bird species classification on edge hardware using TinyML.

## Overview

This project deploys a CNN-based acoustic classifier on a Seeed XIAO ESP32S3 Sense microcontroller to identify California bird species in real-time. The system captures audio via a built-in PDM microphone, extracts Mel-frequency spectrogram features, runs quantized inference on-device, and logs detections to a microSD card.

**Key Metrics:**
- Validation Accuracy: 85%+
- Inference Latency: <500ms per 3-second clip
- Power Consumption: ~150mA active
- Total Hardware Cost: ~$50

## Target Species

| Species | Scientific Name | Distinctive Feature |
|---------|-----------------|---------------------|
| Western Meadowlark | *Sturnella neglecta* | Flute-like melody |
| Red-tailed Hawk | *Buteo jamaicensis* | Raspy scream |
| California Quail | *Callipepla californica* | "Chi-ca-go" call |
| American Crow | *Corvus brachyrhynchos* | Harsh "caw" |
| Great Horned Owl | *Bubo virginianus* | Deep hooting |
| Background/Noise | — | Non-bird sounds |

## Hardware

| Component | Purpose |
|-----------|---------|
| Seeed XIAO ESP32S3 Sense | MCU with built-in mic and SD slot |
| 32GB microSD card (FAT32) | Detection logging |
| USB-C cable (10ft) | Power supply |
| PVC enclosure (DIY) | Weatherproofing |

## Project Structure

```
bird-acoustic-detection/
├── pyproject.toml          # Dependencies (uv)
├── scripts/
│   ├── download_xenocanto.py   # Dataset acquisition
│   ├── download_esc50.py       # Background noise dataset
│   ├── preprocess_audio.py     # Audio normalization
│   ├── augment_audio.py        # Data augmentation
│   ├── dataset.py              # PyTorch dataset class
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training loop
│   └── export_model.py         # ONNX/TFLite export
├── firmware/
│   └── bird_detector/
│       └── bird_detector.ino   # ESP32 firmware
├── models/                     # Trained models and artifacts
├── data/                       # Audio datasets (not committed)
└── docs/                       # Additional documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Arduino IDE 2.x (for firmware deployment)

### Installation

```bash
cd bird-acoustic-detection

# Install dependencies
uv sync

# Verify installation
uv run python scripts/test_setup.py
```

### Training Pipeline

```bash
# 1. Download bird recordings from Xeno-canto
uv run python scripts/download_xenocanto.py

# 2. Download background noise (ESC-50)
uv run python scripts/download_bg.py

# 3. Preprocess audio to 16kHz, 3-second clips
uv run python scripts/preprocess_audio.py

# 4. Apply data augmentation
uv run python scripts/augment_audio.py

# 5. Train model
uv run python scripts/train.py --epochs 50 --batch-size 32

# 6. Export for edge deployment
uv run python scripts/export_model.py
```

### Firmware Deployment

1. Open `firmware/bird_detector/bird_detector.ino` in Arduino IDE
2. Install ESP32 board support (Espressif Systems v3.x)
3. Select board: **XIAO_ESP32S3**
4. Add Edge Impulse library (generated from trained model)
5. Upload to device

## Technical Details

### Model Architecture

Compact CNN optimized for ESP32-S3 constraints:

- 4 convolutional blocks (16 → 32 → 64 → 64 channels)
- Batch normalization and max pooling
- Adaptive average pooling for variable input sizes
- ~25K parameters (~100KB quantized)

### Audio Processing

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16 kHz |
| Clip Duration | 3 seconds |
| FFT Size | 512 |
| Hop Length | 256 |
| Mel Bands | 40 |

### Data Augmentation

- Gaussian noise injection
- Time stretching (0.8x - 1.2x)
- Pitch shifting (±2 semitones)
- Random time shifting
- Gain variation (±6 dB)

## Results

Training on ~1,800 augmented samples (300 per class) typically achieves:

- Training Accuracy: 92-95%
- Validation Accuracy: 85-88%
- Per-class F1 scores: 0.80-0.92

See `models/training_curves.png` and `models/confusion_matrix.png` for detailed results.

## Deployment

### Indoor Testing

Connect XIAO via USB, open Serial Monitor at 115200 baud. The device will:
1. Initialize microphone and SD card
2. Record 3-second audio windows
3. Run inference and print classification results
4. Log detections above confidence threshold to CSV

### Outdoor Installation

1. Build weatherproof enclosure from 4" PVC pipe
2. Drill holes for USB cable entry and microphone sound path
3. Seal cable entry with silicone caulk
4. Mount under eave with microphone facing away from prevailing wind
5. Route USB cable to indoor power source

## Dependencies

Core dependencies managed via `pyproject.toml`:

- PyTorch 2.0+ (CPU)
- torchaudio
- librosa
- audiomentations
- scikit-learn
- matplotlib / seaborn

See `pyproject.toml` for complete list.

## References

- [Xeno-canto](https://xeno-canto.org/) - Bird recording database
- [ESC-50](https://github.com/karolpiczak/ESC-50) - Environmental sound classification dataset
- [Edge Impulse](https://edgeimpulse.com/) - TinyML deployment platform
- [XIAO ESP32S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/) - Hardware documentation

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

arcT

---

*Built as a portfolio project demonstrating end-to-end TinyML development: data collection, model training, quantization, and edge deployment.*
