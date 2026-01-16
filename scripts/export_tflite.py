#!/usr/bin/env python3
"""
export_tflite.py - California Bird Acoustic Detection
Direct PyTorch → TFLite Export using ai-edge-torch

This script REPLACES the broken ONNX→TF→TFLite pipeline.
It uses Google's ai-edge-torch for direct conversion with PT2E quantization.

Usage:
    uv run python scripts/export_tflite.py
    
Requirements:
    - torch>=2.5.0
    - ai-edge-torch>=0.7.0
    - ai-edge-litert>=1.0.0
    
Author: arcT
Date: January 2026
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys
from typing import Tuple, List, Optional

# ai-edge-torch imports
import ai_edge_torch
from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e


# =============================================================================
# Model Architecture (must match training)
# =============================================================================

import torch.nn as nn

class BirdClassifierCNN(nn.Module):
    """
    Compact CNN for bird audio classification.
    Designed for edge deployment on ESP32-S3.
    
    Input: Mel spectrogram (1, n_mels, time_frames) = (1, 40, 188)
    Output: Class logits (num_classes,)
    """
    
    def __init__(self, num_classes: int = 6, n_mels: int = 40):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_name: str = "standard", num_classes: int = 6, n_mels: int = 40) -> nn.Module:
    """Factory function to get model by name."""
    if model_name == "standard":
        return BirdClassifierCNN(num_classes, n_mels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Export Functions
# =============================================================================

def load_model(checkpoint_path: Path) -> Tuple[nn.Module, List[str]]:
    """
    Load trained PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        Tuple of (model, class_names)
    """
    print(f"📂 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract configuration
    classes = checkpoint.get("classes", [
        "american_crow", "background", "california_quail",
        "great_horned_owl", "red_tailed_hawk", "western_meadowlark"
    ])
    num_classes = len(classes)
    
    config = checkpoint.get("config", {"model": "standard"})
    model_type = config.get("model", "standard")
    
    # Create and load model
    model = get_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Report model stats
    param_count = model.count_parameters()
    print(f"   Model type: {model_type}")
    print(f"   Classes: {classes}")
    print(f"   Parameters: {param_count:,}")
    
    # Check for sparsity (LTH pruning)
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    if zero_params > 0:
        sparsity = zero_params / total_params * 100
        print(f"   Sparsity: {sparsity:.1f}% (LTH pruned)")
    
    return model, classes


def create_calibration_data(n_samples: int = 100) -> List[torch.Tensor]:
    """
    Generate calibration data for quantization.
    
    For best results, this should use REAL spectrograms from the dataset.
    Random data is used here as a fallback.
    
    Args:
        n_samples: Number of calibration samples
        
    Returns:
        List of input tensors
    """
    print(f"🔬 Generating {n_samples} calibration samples...")
    
    # Input shape: (batch=1, channels=1, n_mels=40, time_frames=188)
    # This matches 3-second audio @ 16kHz with n_fft=512, hop_length=256
    calibration_data = []
    
    for _ in range(n_samples):
        # Generate random spectrogram-like data
        # In practice, load real spectrograms for better calibration
        sample = torch.randn(1, 1, 40, 188)
        # Normalize like training data
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        calibration_data.append(sample)
    
    return calibration_data


def load_real_calibration_data(data_dir: Path, n_samples: int = 100) -> List[torch.Tensor]:
    """
    Load real spectrograms from processed dataset for calibration.
    This provides better quantization accuracy than random data.
    
    Args:
        data_dir: Path to processed audio directory
        n_samples: Number of samples to load
        
    Returns:
        List of spectrogram tensors
    """
    try:
        import librosa
        import torchaudio.transforms as T
    except ImportError:
        print("   ⚠️  librosa/torchaudio not available, using random calibration data")
        return create_calibration_data(n_samples)
    
    audio_files = list(data_dir.rglob("*.wav"))
    
    if len(audio_files) < n_samples:
        print(f"   ⚠️  Only {len(audio_files)} files available, using all")
        n_samples = len(audio_files)
    
    # Random sample
    import random
    selected_files = random.sample(audio_files, n_samples)
    
    # Setup mel spectrogram transform (same as training)
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=40,
        power=2.0
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=80)
    
    calibration_data = []
    
    for audio_path in selected_files:
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            waveform = torch.from_numpy(y).unsqueeze(0)  # (1, samples)
            
            # Compute mel spectrogram
            mel_spec = mel_transform(waveform)
            mel_spec_db = db_transform(mel_spec)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Ensure correct shape (1, 1, 40, time)
            if mel_spec_db.dim() == 3:
                mel_spec_db = mel_spec_db.unsqueeze(0)
            
            # Pad/truncate time dimension to 188 frames
            target_frames = 188
            if mel_spec_db.shape[-1] < target_frames:
                pad_size = target_frames - mel_spec_db.shape[-1]
                mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad_size))
            else:
                mel_spec_db = mel_spec_db[..., :target_frames]
            
            calibration_data.append(mel_spec_db)
            
        except Exception as e:
            print(f"   ⚠️  Error loading {audio_path}: {e}")
            continue
    
    if len(calibration_data) < 10:
        print("   ⚠️  Not enough real data, falling back to random calibration")
        return create_calibration_data(n_samples)
    
    print(f"   ✅ Loaded {len(calibration_data)} real spectrograms for calibration")
    return calibration_data


def export_float32(model: nn.Module, output_path: Path) -> ai_edge_torch.model.TfLiteModel:
    """
    Export model to TFLite format (float32).
    
    Args:
        model: PyTorch model in eval mode
        output_path: Where to save the .tflite file
        
    Returns:
        Converted edge model for verification
    """
    print("\n" + "=" * 50)
    print("📦 Exporting Float32 Model")
    print("=" * 50)
    
    sample_input = (torch.randn(1, 1, 40, 188),)
    
    print("   Converting PyTorch → TFLite...")
    edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    
    print(f"   Saving to {output_path}...")
    edge_model.export(str(output_path))
    
    size_kb = output_path.stat().st_size / 1024
    print(f"   ✅ Export complete")
    print(f"   📊 Size: {size_kb:.1f} KB")
    
    return edge_model


def export_int8_quantized(
    model: nn.Module,
    output_path: Path,
    calibration_data: List[torch.Tensor]
) -> ai_edge_torch.model.TfLiteModel:
    """
    Export model to TFLite format with int8 quantization using PT2E.
    
    Args:
        model: PyTorch model in eval mode
        output_path: Where to save the .tflite file
        calibration_data: List of input tensors for calibration
        
    Returns:
        Converted edge model for verification
    """
    print("\n" + "=" * 50)
    print("📦 Exporting Int8 Quantized Model (PT2E)")
    print("=" * 50)
    
    sample_input = (torch.randn(1, 1, 40, 188),)
    
    # Setup PT2E quantizer with symmetric int8 configuration
    print("   Setting up PT2E quantizer...")
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True,   # Per-channel for weights (better accuracy)
            is_dynamic=False,       # Static quantization (required for edge)
        )
    )
    
    # Export model to intermediate representation
    print("   Capturing model graph with torch.export...")
    try:
        # PyTorch 2.6+ syntax
        pt2e_model = torch.export.export(model.eval(), sample_input).module()
    except AttributeError:
        # Fallback for older PyTorch
        from torch._export import capture_pre_autograd_graph
        pt2e_model = capture_pre_autograd_graph(model.eval(), sample_input)
    
    # Insert quantization observers
    print("   Preparing quantization observers...")
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)
    
    # Run calibration
    print(f"   Running calibration ({len(calibration_data)} samples)...")
    for i, data in enumerate(calibration_data):
        pt2e_model(data)
        if (i + 1) % 25 == 0:
            print(f"      Calibrated {i + 1}/{len(calibration_data)}")
    
    # Convert to quantized model
    print("   Converting to quantized representation...")
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    
    # Convert to TFLite
    print("   Converting to TFLite format...")
    edge_model = ai_edge_torch.convert(
        pt2e_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    
    print(f"   Saving to {output_path}...")
    edge_model.export(str(output_path))
    
    size_kb = output_path.stat().st_size / 1024
    print(f"   ✅ Export complete")
    print(f"   📊 Size: {size_kb:.1f} KB")
    
    return edge_model


def verify_conversion(
    pytorch_model: nn.Module,
    edge_model,
    n_tests: int = 20
) -> Tuple[float, float]:
    """
    Verify TFLite model matches PyTorch output using cosine similarity.
    
    Args:
        pytorch_model: Original PyTorch model
        edge_model: Converted TFLite model
        n_tests: Number of random inputs to test
        
    Returns:
        Tuple of (average_similarity, minimum_similarity)
    """
    print(f"\n🔬 Verification ({n_tests} test samples)...")
    
    similarities = []
    max_diff = 0.0
    
    for i in range(n_tests):
        # Generate random input
        test_input = torch.randn(1, 1, 40, 188)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy().flatten()
        
        # TFLite inference
        edge_output = np.array(edge_model(test_input)).flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(pytorch_output, edge_output)
        norm_product = np.linalg.norm(pytorch_output) * np.linalg.norm(edge_output)
        cos_sim = dot_product / (norm_product + 1e-8)
        similarities.append(cos_sim)
        
        # Track maximum absolute difference
        abs_diff = np.max(np.abs(pytorch_output - edge_output))
        max_diff = max(max_diff, abs_diff)
        
        if (i + 1) % 5 == 0:
            print(f"   Sample {i + 1}: cos_sim={cos_sim:.6f}")
    
    avg_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    
    print(f"\n   📊 Results:")
    print(f"      Average cosine similarity: {avg_similarity:.6f}")
    print(f"      Minimum cosine similarity: {min_similarity:.6f}")
    print(f"      Maximum absolute difference: {max_diff:.6f}")
    
    return avg_similarity, min_similarity


def generate_c_header(classes: List[str], output_path: Path) -> None:
    """
    Generate C header file with class names for ESP32 firmware.
    
    Args:
        classes: List of class names
        output_path: Where to save bird_classes.h
    """
    header_content = f"""// Auto-generated by export_tflite.py
// California Bird Acoustic Detection System
// Class names for ESP32 firmware

#ifndef BIRD_CLASSES_H
#define BIRD_CLASSES_H

#define NUM_CLASSES {len(classes)}

const char* CLASS_NAMES[NUM_CLASSES] = {{
{chr(10).join(f'    "{cls}",' for cls in classes)}
}};

// Class indices for easy reference
{chr(10).join(f'#define CLASS_{cls.upper()} {i}' for i, cls in enumerate(classes))}

#endif // BIRD_CLASSES_H
"""
    
    output_path.write_text(header_content)
    print(f"   ✅ C header saved to {output_path}")


def generate_model_info(
    model: nn.Module,
    classes: List[str],
    float32_path: Path,
    int8_path: Path,
    output_path: Path
) -> None:
    """
    Generate JSON file with model metadata.
    
    Args:
        model: PyTorch model
        classes: Class names
        float32_path: Path to float32 model
        int8_path: Path to int8 model
        output_path: Where to save model_info.json
    """
    info = {
        "model_name": "BirdClassifierCNN",
        "version": "3.0",
        "export_date": "2026-01-08",
        "classes": classes,
        "num_classes": len(classes),
        "input_shape": [1, 1, 40, 188],
        "input_description": "Mel spectrogram: (batch, channels, n_mels, time_frames)",
        "audio_config": {
            "sample_rate": 16000,
            "duration_seconds": 3.0,
            "n_mels": 40,
            "n_fft": 512,
            "hop_length": 256
        },
        "parameters": model.count_parameters(),
        "models": {
            "float32": {
                "filename": float32_path.name,
                "size_kb": float32_path.stat().st_size / 1024
            },
            "int8": {
                "filename": int8_path.name,
                "size_kb": int8_path.stat().st_size / 1024
            }
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ Model info saved to {output_path}")


# =============================================================================
# Main Export Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("🚀 California Bird Acoustic Detection")
    print("   TFLite Export via ai-edge-torch")
    print("=" * 60)
    
    # Configuration
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    
    # Model paths (try pruned first, then unpruned)
    model_candidates = [
        project_dir / "models" / "lottery_ticket" / "winning_ticket_79pct.pth",
        project_dir / "models" / "best_model.pth",
        project_dir / "models" / "final_model.pth",
    ]
    
    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        print("\n❌ No model checkpoint found!")
        print("   Looked in:")
        for c in model_candidates:
            print(f"      {c}")
        print("\n   Run training first: uv run python scripts/train.py")
        sys.exit(1)
    
    # Output directory
    output_dir = project_dir / "models" / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data directory for calibration
    data_dir = project_dir / "data" / "processed"
    
    # Load model
    model, classes = load_model(model_path)
    
    # Generate calibration data
    if data_dir.exists():
        calibration_data = load_real_calibration_data(data_dir, n_samples=100)
    else:
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("   Using random calibration data (less accurate quantization)")
        calibration_data = create_calibration_data(100)
    
    # Export float32 model
    float32_path = output_dir / "bird_classifier_float32.tflite"
    edge_model_f32 = export_float32(model, float32_path)
    
    # Export int8 quantized model
    int8_path = output_dir / "bird_classifier_int8.tflite"
    edge_model_int8 = export_int8_quantized(model, int8_path, calibration_data)
    
    # Verify conversions
    print("\n" + "=" * 50)
    print("🔬 Verifying Float32 Model")
    print("=" * 50)
    avg_f32, min_f32 = verify_conversion(model, edge_model_f32)
    
    print("\n" + "=" * 50)
    print("🔬 Verifying Int8 Model")
    print("=" * 50)
    avg_int8, min_int8 = verify_conversion(model, edge_model_int8)
    
    # Generate supporting files
    print("\n" + "=" * 50)
    print("📝 Generating Supporting Files")
    print("=" * 50)
    generate_c_header(classes, output_dir / "bird_classes.h")
    generate_model_info(model, classes, float32_path, int8_path, output_dir / "model_info.json")
    
    # Boss fight validation
    print("\n" + "=" * 60)
    print("🎮 BOSS FIGHT: Conversion Verification")
    print("=" * 60)
    
    threshold = 0.99
    
    if min_int8 >= threshold:
        print(f"🏆 BOSS FIGHT PASSED!")
        print(f"   Minimum int8 similarity: {min_int8:.4f} >= {threshold}")
        boss_passed = True
    elif min_int8 >= 0.95:
        print(f"⚠️  BOSS FIGHT MARGINAL")
        print(f"   Minimum int8 similarity: {min_int8:.4f}")
        print("   Consider using real calibration data for better results")
        boss_passed = True
    else:
        print(f"💀 BOSS FIGHT FAILED")
        print(f"   Minimum int8 similarity: {min_int8:.4f} < {threshold}")
        print("   Investigate quantization issues")
        boss_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Export Summary")
    print("=" * 60)
    
    f32_size = float32_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024
    compression = f32_size / int8_size
    
    print(f"   Float32 model: {f32_size:.1f} KB")
    print(f"   Int8 model:    {int8_size:.1f} KB")
    print(f"   Compression:   {compression:.1f}x smaller")
    print(f"\n   Output directory: {output_dir}")
    print(f"   Files created:")
    print(f"      - {float32_path.name}")
    print(f"      - {int8_path.name}")
    print(f"      - bird_classes.h")
    print(f"      - model_info.json")
    
    print("\n" + "=" * 60)
    
    if boss_passed:
        print("✅ Ready for Level 5: Hardware Deployment!")
        print(f"\n   Next steps:")
        print(f"   1. Copy {int8_path.name} to firmware/bird_detector/src/")
        print(f"   2. Convert to C array: xxd -i {int8_path.name} > model_data.h")
        print(f"   3. Build firmware: cd firmware/bird_detector && idf.py build")
    else:
        print("❌ Fix quantization issues before proceeding to deployment")
    
    print("=" * 60)
    
    return 0 if boss_passed else 1


if __name__ == "__main__":
    sys.exit(main())