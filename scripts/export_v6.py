#!/usr/bin/env python3
"""
export_v6.py - Export model using MEAN op for global average pooling

The problem:
- AvgPool2d(5,23) -> AVERAGE_POOL_2D op with unusual kernel = 11.8s on ESP32
- Flatten + FC -> Model too large (546KB) and DEPTHWISE_CONV_2D version issues

The solution:
- Use torch.mean(dim=[2,3]) -> MEAN op in TFLite
- MEAN op is well-optimized for global average pooling
- Keeps model small (~75KB)

Usage:
    uv run python scripts/export_v6.py \
        --model-path models/energy_filtered_v1/best_model.pth \
        --data-dir data/augmented \
        --output-dir models/exported_v2
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np

import ai_edge_torch
from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization import allow_exported_model_train_eval, move_exported_model_to_eval

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_v3 import BirdAudioDatasetV3, EXPECTED_CLASSES_9


# v6 (C1): mel bins — MUST match AlignedMelSpectrogram in dataset_v3.py (was 40, now 64).
# Used for the example/trace input shape (1,1,MODEL_N_MELS,188) throughout export.
MODEL_N_MELS = 64

# =============================================================================
# ESP32-OPTIMIZED MODEL WITH MEAN-BASED GLOBAL POOLING
# =============================================================================

class BirdClassifierESP32Mean(nn.Module):
    """
    ESP32-compatible model using torch.mean() for global average pooling.
    
    torch.mean(dim=[2,3]) converts to MEAN op in TFLite which is well-optimized.
    This avoids the slow AVERAGE_POOL_2D with unusual kernel sizes.
    
    Input: (1, 1, 40, 188)
    After conv layers: (1, 64, 5, 23)
    After mean: (1, 64, 1, 1) -> squeeze -> (1, 64)
    FC: (1, 64) -> (1, 32) -> (1, 9)
    """

    def __init__(self, num_classes: int = 9, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x188 -> 20x94
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 20x94 -> 10x47
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 10x47 -> 5x23
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            # NO pooling - use mean() in forward
        )

        # Same classifier structure as original
        self.classifier = nn.Sequential(
            nn.Linear(c * 4, c * 2),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(c * 2, num_classes),  # 32 -> 9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling using mean
        # This converts to MEAN op in TFLite (well-optimized)
        x = x.mean(dim=[2, 3])  # (B, 64, 5, 23) -> (B, 64)
        
        x = self.classifier(x)
        return x


class BirdClassifierOriginal(nn.Module):
    """Original architecture for loading trained weights."""

    def __init__(self, num_classes: int = 9, base_channels: int = 16):
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
    """
    Transfer weights - handles classifier index differences.
    
    BirdClassifierOriginal classifier: 0=Flatten, 1=Linear, 2=ReLU, 3=Dropout, 4=Linear
    BirdClassifierESP32Mean classifier: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()

    # Mapping for classifier layers (Original -> ESP32Mean)
    classifier_map = {
        "classifier.1.": "classifier.0.",  # First Linear
        "classifier.4.": "classifier.3.",  # Second Linear
    }

    transferred = 0
    for k, v in src_state.items():
        dst_key = k
        
        # Check if this is a classifier key that needs remapping
        for old_prefix, new_prefix in classifier_map.items():
            if k.startswith(old_prefix):
                dst_key = new_prefix + k[len(old_prefix):]
                break
        
        if dst_key in dst_state and dst_state[dst_key].shape == v.shape:
            dst_state[dst_key] = v
            transferred += 1

    dst_model.load_state_dict(dst_state, strict=True)
    print(f"  Transferred {transferred} tensors")


def remap_train_v4_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap train_v4.py checkpoint keys."""
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

    # train_v4 classifier: 0=Flatten, 1=Dropout, 2=Linear, 3=ReLU, 4=Dropout, 5=Linear
    # BirdClassifierOriginal: 0=Flatten, 1=Linear, 2=ReLU, 3=Dropout, 4=Linear
    cls_map = {
        "classifier.2.": "classifier.1.",  # First Linear
        "classifier.5.": "classifier.4.",  # Second Linear
    }

    out = {}
    for k, v in sd.items():
        nk = k
        for a, b in feature_map.items():
            nk = repl_prefix(nk, a, b)
        for a, b in cls_map.items():
            nk = repl_prefix(nk, a, b)
        out[nk] = v

    return out


def force_exported_eval(m: nn.Module) -> nn.Module:
    m = allow_exported_model_train_eval(m)
    m = move_exported_model_to_eval(m)
    m.training = False
    return m


# =============================================================================
# CALIBRATION DATA
# =============================================================================

def generate_calibration_data(data_dir: Path, n_samples: int = 200, seed: int = 42) -> List[torch.Tensor]:
    print(f"\n=== Generating {n_samples} Calibration Samples ===")
    dataset = BirdAudioDatasetV3(str(data_dir), augment=False)

    np.random.seed(seed)
    n_samples = min(n_samples, len(dataset))
    idxs = np.random.choice(len(dataset), n_samples, replace=False)

    data = []
    for i in idxs:
        spec, _ = dataset[i]
        data.append(spec.unsqueeze(0))

    all_specs = torch.cat(data, dim=0)
    print(f"  Shape: {all_specs.shape}")
    print(f"  Mean: {all_specs.mean():.4f}, Std: {all_specs.std():.4f}")

    return data


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_float32(model: nn.Module, output_path: Path) -> None:
    print("\n=== Exporting Float32 Model ===")
    model.eval()
    sample_input = torch.randn(1, 1, MODEL_N_MELS, 188)

    edge_model = ai_edge_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB)")


def export_int8(model: nn.Module, output_path: Path, calibration_data: List[torch.Tensor]) -> None:
    print("\n=== Exporting Int8 Model ===")

    model.eval()
    sample_input = (torch.randn(1, 1, MODEL_N_MELS, 188),)

    print("  Configuring PT2E quantizer...")
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )

    print("  Exporting model graph...")
    try:
        pt2e_model = torch.export.export(model, sample_input).module()
    except AttributeError:
        from torch._export import capture_pre_autograd_graph
        pt2e_model = capture_pre_autograd_graph(model, sample_input)

    print("  Inserting quantization observers...")
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)
    pt2e_model = force_exported_eval(pt2e_model)

    print(f"  Running calibration ({len(calibration_data)} samples)...")
    with torch.no_grad():
        for i, x in enumerate(calibration_data):
            pt2e_model(x)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(calibration_data)}")

    print("  Converting to quantized model...")
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    pt2e_model = force_exported_eval(pt2e_model)

    print("  Exporting to TFLite...")
    edge_model = ai_edge_torch.convert(
        pt2e_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )
    edge_model.export(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB)")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_model(tflite_path: Path, pytorch_model: nn.Module, dataset: BirdAudioDatasetV3, n_samples: int = 100) -> Dict:
    print("\n=== Verifying TFLite Model ===")

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("  ai_edge_litert not available")
        return {}

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"  Input:  dtype={input_details['dtype']}, shape={input_details['shape']}")
    print(f"  Output: dtype={output_details['dtype']}, shape={output_details['shape']}")

    is_quantized = input_details["dtype"] == np.int8
    
    if is_quantized:
        in_scale = input_details["quantization_parameters"]["scales"][0]
        in_zp = int(input_details["quantization_parameters"]["zero_points"][0])
        out_scale = output_details["quantization_parameters"]["scales"][0]
        out_zp = int(output_details["quantization_parameters"]["zero_points"][0])
        print(f"  Input quant:  scale={in_scale:.6f}, zp={in_zp}")
        print(f"  Output quant: scale={out_scale:.6f}, zp={out_zp}")
        print("  ✓ Model is int8 quantized")
    else:
        print("  ⚠ Model is float32")

    np.random.seed(42)
    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    pytorch_model.eval()
    pt_correct = 0
    tfl_correct = 0
    agreement = 0

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

    results = {
        "pytorch_accuracy": 100.0 * pt_correct / n_samples,
        "tflite_accuracy": 100.0 * tfl_correct / n_samples,
        "agreement": 100.0 * agreement / n_samples,
        "is_quantized": is_quantized,
    }

    print(f"\n  PyTorch accuracy:  {results['pytorch_accuracy']:.1f}%")
    print(f"  TFLite accuracy:   {results['tflite_accuracy']:.1f}%")
    print(f"  Agreement:         {results['agreement']:.1f}%")

    return results


def inspect_tflite_ops(tflite_path: Path) -> List[str]:
    """List all ops used in the TFLite model."""
    print("\n=== Inspecting TFLite Ops ===")
    
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    # Try to use flatbuffers to parse
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get op details
        ops = interpreter._get_ops_details()
        op_names = set()
        for op in ops:
            op_names.add(op['op_name'])
        
        print(f"  Ops used: {sorted(op_names)}")
        return sorted(op_names)
    except Exception as e:
        print(f"  Could not inspect ops: {e}")
        return []


# =============================================================================
# HEADER GENERATION
# =============================================================================

def generate_header(tflite_path: Path, header_path: Path, array_name: str) -> None:
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("// Auto-generated TFLite model header — do not edit\n")
        f.write(f"// Source: {tflite_path.name}\n")
        f.write(f"// Size: {len(data)} bytes ({len(data)/1024:.1f} KB)\n")
        f.write("// Architecture: MEAN-based global pooling (ESP32 optimized)\n")
        f.write("#pragma once\n\n")
        # 16-byte alignment: TFLM requires an aligned flatbuffer; lives in flash .rodata
        f.write(f"__attribute__((aligned(16))) const unsigned char {array_name}[] = {{\n")

        for i in range(0, len(data), 12):
            chunk = ", ".join(f"0x{b:02x}" for b in data[i:i + 12])
            f.write(f"    {chunk},\n")

        f.write("};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")

    print(f"  Saved: {header_path} ({len(data)/1024:.1f} KB)")


def generate_model_meta_header(
    tflite_path: Path,
    classes: List[str],
    calibration: Dict,
    header_path: Path,
    n_mels: int = 64,       # v6 (C1): must match AlignedMelSpectrogram in dataset_v3.py
    f_max: float = 7000.0,  # v6 (C1): documented here; realized in frontend_tables.h
) -> None:
    """model_meta.h: quantization params + decision rule + frontend contract.
    Firmware includes this and hardcodes NO magic numbers."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale = float(inp["quantization_parameters"]["scales"][0])
    in_zp = int(inp["quantization_parameters"]["zero_points"][0])
    out_scale = float(out["quantization_parameters"]["scales"][0])
    out_zp = int(out["quantization_parameters"]["zero_points"][0])

    with open(header_path, "w") as f:
        f.write("// Auto-generated model metadata — do not edit\n")
        f.write(f"// Source: {tflite_path.name}\n")
        f.write("#pragma once\n\n")
        f.write("// --- int8 quantization ---\n")
        f.write(f"#define MODEL_INPUT_SCALE   {in_scale:.9e}f\n")
        f.write(f"#define MODEL_INPUT_ZP      {in_zp}\n")
        f.write(f"#define MODEL_OUTPUT_SCALE  {out_scale:.9e}f\n")
        f.write(f"#define MODEL_OUTPUT_ZP     {out_zp}\n\n")
        f.write("// --- decision rule: p = softmax(logits/T); detect iff\n")
        f.write("//     argmax != BACKGROUND_IDX && p[argmax] >= DETECT_TAU_PER_CLASS[argmax] ---\n")
        f.write(f"#define DETECT_TEMPERATURE  {calibration['temperature']:.6f}f\n")
        f.write(f"#define DETECT_TAU          {calibration['tau']:.6f}f  // global fallback / legacy\n")
        f.write(f"#define BACKGROUND_IDX      {classes.index('background')}\n\n")
        f.write("// --- audio frontend contract (must match training exactly) ---\n")
        f.write("#define AUDIO_SAMPLE_RATE   16000\n")
        f.write("#define CLIP_SAMPLES        48000   // 3.0 s\n")
        f.write("#define N_FFT               512\n")
        f.write("#define HOP_LENGTH          256\n")
        f.write(f"#define N_MELS              {n_mels}\n")
        f.write("#define N_FRAMES            188     // 1 + CLIP_SAMPLES/HOP (center=true)\n")
        f.write("#define N_FFT_BINS          257     // N_FFT/2 + 1\n")
        f.write(f"#define MEL_F_MAX           {f_max:.1f}f  // realized in frontend_tables.h\n")
        f.write("#define TOP_DB              80.0f   // dB clamp below global max\n\n")
        f.write(f"#define N_CLASSES           {len(classes)}\n")
        f.write("static const char *const CLASS_NAMES[N_CLASSES] = {\n")
        for cls in classes:
            f.write(f'    "{cls}",\n')
        f.write("};\n\n")
        # v6: per-class detection thresholds (index by argmax). background entry is unused
        # (background is excluded from detection); fall back to the scalar tau if absent.
        tpc = calibration.get("tau_per_class") or [calibration["tau"]] * len(classes)
        f.write("// per-class softmax detection thresholds; index by argmax\n")
        f.write("static const float DETECT_TAU_PER_CLASS[N_CLASSES] = {\n")
        for cls, t in zip(classes, tpc):
            f.write(f"    {float(t):.6f}f,  // {cls}\n")
        f.write("};\n")

    print(f"  Saved: {header_path} (in {in_scale:.6f}/{in_zp}, out {out_scale:.6f}/{out_zp}, "
          f"T={calibration['temperature']:.3f}, tau={calibration['tau']:.2f}, n_mels={n_mels})")


def generate_frontend_tables_header(header_path: Path, n_mels: int = 64,
                                    f_max: float = 7000.0) -> None:
    """frontend_tables.h: torchaudio's own mel filterbank (sparse rows) + periodic
    Hann window, so device mel/window parity holds by construction.
    n_mels / f_max MUST match AlignedMelSpectrogram in dataset_v3.py (v6 C1: 64 / 7000)."""
    import torchaudio.transforms as T

    mel = T.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=256, n_mels=n_mels,
        power=2.0, center=True, norm="slaney", mel_scale="htk", f_max=f_max,
    )
    fb = mel.mel_scale.fb.numpy()                          # (257, n_mels)
    hann = torch.hann_window(512, periodic=True).numpy()   # (512,)

    starts, lens, offsets, weights = [], [], [], []
    for m in range(fb.shape[1]):
        col = fb[:, m]
        nz = np.nonzero(col)[0]
        start = int(nz[0]) if len(nz) else 0
        length = int(nz[-1] - nz[0] + 1) if len(nz) else 0
        starts.append(start)
        lens.append(length)
        offsets.append(len(weights))
        weights.extend(col[start:start + length].tolist())

    def write_array(f, ctype, name, vals, fmt, per_line=8):
        f.write(f"static const {ctype} {name}[{len(vals)}] = {{\n")
        for i in range(0, len(vals), per_line):
            f.write("    " + ", ".join(fmt(v) for v in vals[i:i + per_line]) + ",\n")
        f.write("};\n\n")

    with open(header_path, "w") as f:
        f.write("// Auto-generated frontend tables (exact torchaudio tensors) — do not edit\n")
        f.write("// mel[m] = sum_k MEL_FB_WEIGHTS[MEL_FB_OFFSET[m] + k] * power[MEL_FB_START[m] + k]\n")
        f.write("#pragma once\n\n")
        write_array(f, "short", "MEL_FB_START", starts, lambda v: str(v), 16)
        write_array(f, "short", "MEL_FB_LEN", lens, lambda v: str(v), 16)
        write_array(f, "int", "MEL_FB_OFFSET", offsets, lambda v: str(v), 16)
        write_array(f, "float", "MEL_FB_WEIGHTS", weights, lambda v: f"{v:.8e}f")
        write_array(f, "float", "HANN_WINDOW", hann.tolist(), lambda v: f"{v:.8e}f")

    print(f"  Saved: {header_path} ({len(weights)} filterbank weights, 512 window taps)")


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

    print(f"  Saved: {header_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Export model with MEAN-based global pooling")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    parser.add_argument("--output-dir", type=str, default="models/exported")
    parser.add_argument("--n-calibration", type=int, default=200)
    parser.add_argument("--calibration-json", type=str, default=None,
                        help="Default: calibration.json next to --model-path")
    args = parser.parse_args()

    print("=" * 60)
    print("TFLite Export with MEAN-based Global Pooling (v6)")
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

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes", EXPECTED_CLASSES_9)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Load original model
    original_model = BirdClassifierOriginal(num_classes=num_classes)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("features.") for k in state_dict.keys()):
        print("Remapping train_v4 checkpoint keys...")
        state_dict = remap_train_v4_state_dict(state_dict)

    original_model.load_state_dict(state_dict, strict=True)
    original_model.eval()
    print("✓ Original model loaded")

    # Create ESP32-optimized model with MEAN pooling
    print("\nCreating ESP32-optimized model (MEAN pooling)...")
    esp32_model = BirdClassifierESP32Mean(num_classes=num_classes)
    transfer_weights(original_model, esp32_model)
    esp32_model.eval()

    # Verify output match
    test_input = torch.randn(1, 1, MODEL_N_MELS, 188)
    with torch.no_grad():
        orig_out = original_model(test_input)
        esp32_out = esp32_model(test_input)
    
    diff = torch.abs(orig_out - esp32_out).max().item()
    print(f"  Max output difference: {diff:.6f}")
    
    if diff > 0.01:
        print("  ⚠ Warning: Outputs differ significantly")
    else:
        print("  ✓ Outputs match")

    # Calibration data
    calibration_data = generate_calibration_data(data_dir, n_samples=args.n_calibration)

    # Export float32
    float32_path = output_dir / "bird_classifier_float32.tflite"
    export_float32(esp32_model, float32_path)
    
    # Inspect ops
    inspect_tflite_ops(float32_path)

    # Export int8
    int8_path = output_dir / "bird_classifier_int8.tflite"
    export_int8(esp32_model, int8_path, calibration_data)

    # Verify
    dataset = BirdAudioDatasetV3(str(data_dir), augment=False)
    results = verify_model(int8_path, esp32_model, dataset)

    # Generate headers
    print("\n=== Generating Headers ===")
    generate_header(int8_path, output_dir / "model_data.h", "bird_classifier_int8_tflite")
    generate_classes_header(classes, output_dir / "bird_classes.h")

    calib_path = Path(args.calibration_json) if args.calibration_json \
        else model_path.parent / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            calibration = json.load(f)
    else:
        print(f"  ⚠ {calib_path} not found — using uncalibrated defaults (T=1, tau=0.85)")
        calibration = {"temperature": 1.0, "tau": 0.85}
    generate_model_meta_header(int8_path, classes, calibration, output_dir / "model_meta.h")
    generate_frontend_tables_header(output_dir / "frontend_tables.h")

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)

    f32_size = float32_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024

    print(f"\nFiles in {output_dir}:")
    print(f"  - bird_classifier_float32.tflite ({f32_size:.1f} KB)")
    print(f"  - bird_classifier_int8.tflite ({int8_size:.1f} KB)")
    print(f"  - model_data.h")
    print(f"  - bird_classes.h")
    
    print(f"\nSize ratio (f32/int8): {f32_size/int8_size:.1f}x")
    
    # Size check
    if int8_size > 150:
        print(f"\n⚠ WARNING: Model size ({int8_size:.0f} KB) is larger than expected")
        print("  Expected ~75 KB for this architecture")
    else:
        print(f"\n✓ Model size looks good ({int8_size:.0f} KB)")
    
    if results:
        print(f"\nVerification:")
        print(f"  - TFLite accuracy: {results['tflite_accuracy']:.1f}%")
        print(f"  - Agreement: {results['agreement']:.1f}%")

    # Save metadata
    metadata = {
        "architecture": "BirdClassifierESP32Mean (torch.mean for global pooling)",
        "classes": classes,
        "expected_esp32_ops": ["CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "QUANTIZE", "DEQUANTIZE"],
        "results": results,
    }
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ This model uses torch.mean() -> MEAN op (should be fast on ESP32)")
    print("  Expected invoke time: <200ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())