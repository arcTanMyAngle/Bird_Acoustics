#!/usr/bin/env python3
"""
export_v2.py - Export trained model to TFLite with proper calibration

Key features:
1) Uses REAL spectrogram data for int8 calibration
2) ESP32-compatible architecture (fixed AvgPool2d instead of AdaptiveAvgPool2d)
3) Verification against PyTorch on a sample set
4) Generates Arduino-ready header files

Usage:
  uv run python scripts/export_v2.py \
    --model-path models/v3/best_model.pth \
    --data-dir data/augmented \
    --output-dir models/exported_v3
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np

# ai-edge-torch imports
import ai_edge_torch
from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

# PT2E exported model train/eval helpers (required: exported GraphModules block .eval/.train)
from torch.ao.quantization import allow_exported_model_train_eval, move_exported_model_to_eval

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_v2 import BirdAudioDatasetV2


# =============================================================================
# ESP32-COMPATIBLE MODEL ARCHITECTURE
# =============================================================================

class BirdClassifierESP32(nn.Module):
    """
    ESP32-compatible model with fixed AvgPool2d instead of AdaptiveAvgPool2d.

    Input: (1, 1, 40, 188)
    After conv layers: (1, 64, 5, 23)
    AvgPool2d(5, 23) -> (1, 64, 1, 1)
    """

    def __init__(self, num_classes: int = 6, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # ESP32-compatible pooling: fixed kernel for known feature map size (5, 23)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(5, 23)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


class BirdClassifierOriginal(nn.Module):
    """Original architecture with AdaptiveAvgPool2d for loading trained weights."""

    def __init__(self, num_classes: int = 6, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


def transfer_weights(src_model: nn.Module, dst_model: nn.Module) -> None:
    """Transfer weights between models with same conv/bn/classifier params (pooling differs)."""
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()

    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k] = v

    dst_model.load_state_dict(dst_state, strict=True)


def remap_train_v3_state_dict_to_export_v2(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap train_v3.py checkpoints:
      - features.*  -> conv1..conv4 blocks
      - classifier.2/5 -> classifier.1/4
    """

    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    def repl_prefix(k: str, old: str, new: str) -> str:
        return new + k[len(old):] if k.startswith(old) else k

    feature_map = {
        "features.0.":  "conv1.0.",
        "features.1.":  "conv1.1.",
        "features.4.":  "conv2.0.",
        "features.5.":  "conv2.1.",
        "features.8.":  "conv3.0.",
        "features.9.":  "conv3.1.",
        "features.12.": "conv4.0.",
        "features.13.": "conv4.1.",
    }

    cls_map = {
        "classifier.2.": "classifier.1.",
        "classifier.5.": "classifier.4.",
    }

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        for a, b in feature_map.items():
            nk = repl_prefix(nk, a, b)
        for a, b in cls_map.items():
            nk = repl_prefix(nk, a, b)
        out[nk] = v

    return out


def force_exported_eval(m: nn.Module) -> nn.Module:
    """
    Exported PT2E GraphModules block .eval()/.train().
    These helpers switch the internal flags for special ops (dropout/bn) and
    also make downstream conversion treat it as eval.
    """
    m = allow_exported_model_train_eval(m)
    m = move_exported_model_to_eval(m)
    # Some converters still key off this flag:
    m.training = False
    return m


# =============================================================================
# CALIBRATION DATA GENERATION
# =============================================================================

def generate_real_calibration_data(data_dir: Path, n_samples: int = 200, seed: int = 42) -> List[torch.Tensor]:
    print("\n=== Generating Calibration Data from Real Spectrograms ===")
    dataset = BirdAudioDatasetV2(str(data_dir), augment=False)

    np.random.seed(seed)
    n_samples = min(n_samples, len(dataset))
    idxs = np.random.choice(len(dataset), n_samples, replace=False)

    data: List[torch.Tensor] = []
    for i in idxs:
        spec, _ = dataset[i]
        data.append(spec.unsqueeze(0))  # add batch

    all_specs = torch.cat(data, dim=0)
    print(f"Calibration samples: {len(data)}")
    print(f"Shape: {all_specs.shape}")
    print("Statistics:")
    print(f"  Mean: {all_specs.mean().item():.4f}")
    print(f"  Std:  {all_specs.std().item():.4f}")
    print(f"  Min:  {all_specs.min().item():.4f}")
    print(f"  Max:  {all_specs.max().item():.4f}")

    return data


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_float32(model: nn.Module, output_path: Path) -> None:
    print("\n=== Exporting Float32 Model ===")
    model.eval()
    sample_input = torch.randn(1, 1, 40, 188)

    edge_model = ai_edge_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path} ({size_kb:.1f} KB)")


def export_int8(model: nn.Module, output_path: Path, calibration_data: List[torch.Tensor]) -> None:
    print("\n=== Exporting Int8 Model ===")

    model.eval()
    sample_input = (torch.randn(1, 1, 40, 188),)

    # Setup quantizer
    print("Configuring PT2E quantizer...")
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )

    # Export to graph
    print("Exporting model graph...")
    try:
        pt2e_model = torch.export.export(model, sample_input).module()
    except AttributeError:
        from torch._export import capture_pre_autograd_graph
        pt2e_model = capture_pre_autograd_graph(model, sample_input)

    # Prepare observers
    print("Inserting quantization observers...")
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)
    pt2e_model = force_exported_eval(pt2e_model)

    # Calibration
    print(f"Running calibration with {len(calibration_data)} samples...")
    with torch.no_grad():
        for i, x in enumerate(calibration_data):
            pt2e_model(x)
            if (i + 1) % 50 == 0:
                print(f"  Calibrated: {i + 1}/{len(calibration_data)}")

    # Convert
    print("Converting to quantized model...")
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    pt2e_model = force_exported_eval(pt2e_model)

    # Export to TFLite
    print("Exporting to TFLite format...")
    edge_model = ai_edge_torch.convert(
        pt2e_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )
    edge_model.export(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path} ({size_kb:.1f} KB)")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_tflite_model(tflite_path: Path, pytorch_model: nn.Module, dataset: BirdAudioDatasetV2, n_samples: int = 100) -> Dict:
    print("\n=== Verifying TFLite Model ===")

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("ai_edge_litert not available, skipping verification")
        return {}

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Input:  dtype={input_details['dtype']}, shape={input_details['shape']}")
    print(f"Output: dtype={output_details['dtype']}, shape={output_details['shape']}")

    is_quantized = input_details["dtype"] == np.int8

    if is_quantized:
        in_scale = input_details["quantization_parameters"]["scales"][0]
        in_zp = int(input_details["quantization_parameters"]["zero_points"][0])
        out_scale = output_details["quantization_parameters"]["scales"][0]
        out_zp = int(output_details["quantization_parameters"]["zero_points"][0])
        print(f"Input quant:  scale={in_scale:.6f}, zp={in_zp}")
        print(f"Output quant: scale={out_scale:.6f}, zp={out_zp}")

    np.random.seed(42)
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    pytorch_model.eval()

    pt_correct = 0
    tfl_correct = 0
    agreement = 0
    cosine_sims = []

    for idx in indices:
        spec, label = dataset[idx]
        spec_np = spec.unsqueeze(0).numpy()

        with torch.no_grad():
            pt_out = pytorch_model(spec.unsqueeze(0)).numpy().flatten()
        pt_pred = int(np.argmax(pt_out))

        if is_quantized:
            spec_q = np.round(spec_np / in_scale + in_zp).clip(-128, 127).astype(np.int8)
            interpreter.set_tensor(input_details["index"], spec_q)
        else:
            interpreter.set_tensor(input_details["index"], spec_np.astype(np.float32))

        interpreter.invoke()

        tfl_out = interpreter.get_tensor(output_details["index"]).flatten()
        if is_quantized:
            tfl_out = (tfl_out.astype(np.float32) - out_zp) * out_scale

        tfl_pred = int(np.argmax(tfl_out))

        if pt_pred == int(label):
            pt_correct += 1
        if tfl_pred == int(label):
            tfl_correct += 1
        if pt_pred == tfl_pred:
            agreement += 1

        cos_sim = float(np.dot(pt_out, tfl_out) / (np.linalg.norm(pt_out) * np.linalg.norm(tfl_out) + 1e-8))
        cosine_sims.append(cos_sim)

    results = {
        "pytorch_accuracy": 100.0 * pt_correct / n_samples,
        "tflite_accuracy": 100.0 * tfl_correct / n_samples,
        "agreement": 100.0 * agreement / n_samples,
        "accuracy_drop": 100.0 * (pt_correct - tfl_correct) / n_samples,
        "mean_cosine_sim": float(np.mean(cosine_sims)),
        "min_cosine_sim": float(np.min(cosine_sims)),
    }

    print(f"\nResults on {n_samples} samples:")
    print(f"  PyTorch accuracy:      {results['pytorch_accuracy']:.1f}%")
    print(f"  TFLite accuracy:       {results['tflite_accuracy']:.1f}%")
    print(f"  Prediction agreement:  {results['agreement']:.1f}%")
    print(f"  Accuracy drop:         {results['accuracy_drop']:.1f}%")
    print(f"  Mean cosine similarity: {results['mean_cosine_sim']:.4f}")
    print(f"  Min cosine similarity:  {results['min_cosine_sim']:.4f}")

    if results["accuracy_drop"] > 5:
        print("  WARNING: >5% accuracy drop from quantization")
    elif results["accuracy_drop"] > 2:
        print("  CAUTION: 2-5% accuracy drop")
    else:
        print("  ✓ Quantization quality looks good")

    return results


# =============================================================================
# HEADER GENERATION
# =============================================================================

def generate_header(tflite_path: Path, header_path: Path, array_name: str) -> None:
    print("\n=== Generating C Header ===")

    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("// Auto-generated TFLite model header\n")
        f.write(f"// Source: {tflite_path.name}\n")
        f.write(f"// Size: {len(data)} bytes ({len(data)/1024:.1f} KB)\n")
        f.write("// Generated with REAL calibration data\n\n")

        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")

        f.write(f"const unsigned char {array_name}[] PROGMEM = {{\n")

        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x}")
            if i < len(data) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n")

        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")

    print(f"Saved: {header_path}")


def generate_classes_header(classes: List[str], header_path: Path) -> None:
    with open(header_path, "w") as f:
        f.write("// Bird class names\n")
        f.write("#ifndef BIRD_CLASSES_H\n")
        f.write("#define BIRD_CLASSES_H\n\n")

        f.write(f"#define N_CLASSES {len(classes)}\n\n")

        f.write("const char* CLASS_NAMES[] = {\n")
        for cls in classes:
            f.write(f'    "{cls}",\n')
        f.write("};\n\n")

        f.write("#endif // BIRD_CLASSES_H\n")

    print(f"Saved: {header_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained model to TFLite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--data-dir", type=str, default="data/augmented", help="Path to data for calibration")
    parser.add_argument("--output-dir", type=str, default="models/exported", help="Output directory for exported models")
    parser.add_argument("--n-calibration", type=int, default=200, help="Number of calibration samples")
    parser.add_argument("--skip-float32", action="store_true", help="Skip float32 export")
    args = parser.parse_args()

    print("=" * 60)
    print("TFLite Export with Real Calibration")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    classes = checkpoint.get(
        "classes",
        [
            "american_crow",
            "background",
            "california_quail",
            "great_horned_owl",
            "red_tailed_hawk",
            "western_meadowlark",
        ],
    )
    num_classes = len(classes)
    print(f"Classes: {classes}")

    # Load trained weights into original architecture
    original_model = BirdClassifierOriginal(num_classes=num_classes)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("features.") for k in state_dict.keys()):
        print("Detected train_v3 checkpoint layout (features.*). Remapping keys for export_v2...")
        state_dict = remap_train_v3_state_dict_to_export_v2(state_dict)

    original_model.load_state_dict(state_dict, strict=True)
    original_model.eval()
    print("Model loaded successfully!")

    # Build ESP32-compatible model and transfer weights
    esp32_model = BirdClassifierESP32(num_classes=num_classes)
    transfer_weights(original_model, esp32_model)
    esp32_model.eval()

    # Sanity check: outputs should match closely
    test_input = torch.randn(1, 1, 40, 188)
    with torch.no_grad():
        orig_out = original_model(test_input)
        esp_out = esp32_model(test_input)
    diff = torch.abs(orig_out - esp_out).max().item()
    print(f"Max output difference after weight transfer: {diff:.6f}")

    # Calibration data (real spectrograms)
    calibration_data = generate_real_calibration_data(data_dir, n_samples=args.n_calibration)

    # Export float32
    if not args.skip_float32:
        float32_path = output_dir / "bird_classifier_float32.tflite"
        export_float32(esp32_model, float32_path)

    # Export int8
    int8_path = output_dir / "bird_classifier_int8.tflite"
    export_int8(esp32_model, int8_path, calibration_data)

    # Verify
    dataset = BirdAudioDatasetV2(str(data_dir), augment=False)
    results = verify_tflite_model(int8_path, esp32_model, dataset)

    # Headers
    generate_header(int8_path, output_dir / "model_data.h", "bird_classifier_int8_tflite")
    generate_classes_header(classes, output_dir / "bird_classes.h")

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)

    print(f"\nFiles created in {output_dir}:")
    if not args.skip_float32:
        f32 = output_dir / "bird_classifier_float32.tflite"
        print(f"  - {f32.name} ({f32.stat().st_size / 1024:.1f} KB)")
    print(f"  - {int8_path.name} ({int8_path.stat().st_size / 1024:.1f} KB)")
    print("  - model_data.h")
    print("  - bird_classes.h")

    if results:
        print("\nQuantization quality:")
        print(f"  - TFLite accuracy: {results['tflite_accuracy']:.1f}%")
        print(f"  - Accuracy drop: {results['accuracy_drop']:.1f}%")

    metadata = {
        "source_model": str(model_path),
        "classes": classes,
        "calibration_samples": args.n_calibration,
        "verification_results": results,
    }
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
