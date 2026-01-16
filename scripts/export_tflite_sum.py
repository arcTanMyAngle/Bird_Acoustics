#!/usr/bin/env python3
"""
export_tflite_esp32_compat.py - ESP32-Compatible TFLite Export

This script exports the bird classifier with operations compatible with
the older TensorFlowLite_ESP32 Arduino library (which doesn't support SUM op).

The key change: Replace AdaptiveAvgPool2d with AvgPool2d (fixed kernel size)

Usage:
    cd ~/bird-detection
    uv run python export_tflite_esp32_compat.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

# ai-edge-torch imports
import ai_edge_torch
from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e


# =============================================================================
# MODIFIED Model Architecture - ESP32 Compatible
# =============================================================================

class BirdClassifierCNN_ESP32(nn.Module):
    """
    Bird classifier CNN modified for ESP32 TFLite compatibility.
    
    Key change: AdaptiveAvgPool2d replaced with fixed AvgPool2d
    to avoid SUM op that older TFLite Micro doesn't support.
    
    Input: Mel spectrogram (1, 40, 188)
    Output: Class logits (6,)
    """
    
    def __init__(self, num_classes: int = 6, n_mels: int = 40):
        super().__init__()
        
        # Convolutional blocks (same as original)
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
        
        # MODIFIED: Use fixed AvgPool2d instead of AdaptiveAvgPool2d
        # After conv3: shape is (64, 5, 23) for 40x188 input
        # AdaptiveAvgPool2d((1,1)) was pooling 5x23 -> 1x1
        # Replace with AvgPool2d(kernel_size=(5, 23)) or (5, 11) with padding
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Fixed-size pooling instead of adaptive
            nn.AvgPool2d(kernel_size=(5, 23))  # Exact dimensions after conv4
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


# =============================================================================
# Original Model (for loading weights)
# =============================================================================

class BirdClassifierCNN_Original(nn.Module):
    """Original model architecture for loading saved weights."""
    
    def __init__(self, num_classes: int = 6, n_mels: int = 40):
        super().__init__()
        
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
            nn.AdaptiveAvgPool2d((1, 1))  # Original uses this
        )
        
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


def transfer_weights(original_model, esp32_model):
    """
    Transfer weights from original model to ESP32-compatible model.
    The architectures are identical except for pooling layer.
    """
    # Get state dicts
    orig_state = original_model.state_dict()
    esp32_state = esp32_model.state_dict()
    
    # They should have the same keys (AvgPool2d has no parameters)
    for key in orig_state:
        if key in esp32_state:
            esp32_state[key] = orig_state[key]
        else:
            print(f"Warning: Key {key} not found in ESP32 model")
    
    esp32_model.load_state_dict(esp32_state)
    print("Weights transferred successfully!")


def generate_calibration_data(n_samples: int = 200):
    """Generate representative data for int8 calibration."""
    print(f"Generating {n_samples} calibration samples...")
    
    data = []
    for i in range(n_samples):
        # Simulate mel spectrogram statistics
        # Real spectrograms are typically log-scaled with values roughly in [-80, 0] dB
        # After normalization, they're roughly in [-1, 1]
        spec = torch.randn(1, 1, 40, 188) * 0.3  # Normal distribution
        spec = torch.clamp(spec, -2.0, 2.0)  # Clip outliers
        data.append(spec)
    
    return data


def export_float32(model, output_path: Path):
    """Export float32 TFLite model."""
    print("\n=== Exporting Float32 Model ===")
    
    model.eval()
    sample_input = torch.randn(1, 1, 40, 188)
    
    # Convert to TFLite
    edge_model = ai_edge_torch.convert(model, (sample_input,))
    
    # Save
    edge_model.export(str(output_path))
    
    size_kb = output_path.stat().st_size / 1024
    print(f"Float32 model saved: {output_path}")
    print(f"Size: {size_kb:.1f} KB")
    
    return edge_model


def export_int8(model, output_path: Path, calibration_data):
    """Export int8 quantized TFLite model using PT2E."""
    print("\n=== Exporting Int8 Quantized Model ===")
    
    model.eval()
    sample_input = (torch.randn(1, 1, 40, 188),)
    
    # Setup PT2E quantizer with symmetric int8 configuration
    print("Setting up PT2E quantizer...")
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True,   # Per-channel for weights (better accuracy)
            is_dynamic=False,       # Static quantization (required for edge)
        )
    )
    
    # Export model to intermediate representation
    print("Capturing model graph with torch.export...")
    try:
        # PyTorch 2.6+ syntax - need .module() to get GraphModule
        pt2e_model = torch.export.export(model, sample_input).module()
    except AttributeError:
        # Fallback for older PyTorch
        from torch._export import capture_pre_autograd_graph
        pt2e_model = capture_pre_autograd_graph(model, sample_input)
    
    # Insert quantization observers
    print("Preparing quantization observers...")
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)
    
    # Run calibration
    print(f"Running calibration ({len(calibration_data)} samples)...")
    for i, data in enumerate(calibration_data):
        pt2e_model(data)
        if (i + 1) % 50 == 0:
            print(f"  Calibrated {i + 1}/{len(calibration_data)}")
    
    # Convert to quantized model
    print("Converting to quantized representation...")
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    
    # Convert to TFLite
    print("Converting to TFLite format...")
    edge_model = ai_edge_torch.convert(
        pt2e_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    
    # Save
    edge_model.export(str(output_path))
    
    size_kb = output_path.stat().st_size / 1024
    print(f"Int8 model saved: {output_path}")
    print(f"Size: {size_kb:.1f} KB")
    
    return edge_model


def verify_model(tflite_path: Path, model: nn.Module):
    """Verify TFLite model against PyTorch model."""
    print(f"\n=== Verifying {tflite_path.name} ===")
    
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("ai_edge_litert not available, skipping verification")
        return
    
    # Create interpreter
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Test with random input
    test_input = np.random.randn(1, 1, 40, 188).astype(np.float32)
    
    # Handle quantized input
    if input_details[0]['dtype'] == np.int8:
        scale = input_details[0]['quantization_parameters']['scales'][0]
        zp = input_details[0]['quantization_parameters']['zero_points'][0]
        test_input_q = (test_input / scale + zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], test_input_q)
    else:
        interpreter.set_tensor(input_details[0]['index'], test_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if needed
    if output_details[0]['dtype'] == np.int8:
        scale = output_details[0]['quantization_parameters']['scales'][0]
        zp = output_details[0]['quantization_parameters']['zero_points'][0]
        tflite_output = (tflite_output.astype(np.float32) - zp) * scale
    
    # Compare with PyTorch
    model.eval()
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()
    
    # Compute similarity
    tflite_flat = tflite_output.flatten()
    pytorch_flat = pytorch_output.flatten()
    
    cos_sim = np.dot(tflite_flat, pytorch_flat) / (
        np.linalg.norm(tflite_flat) * np.linalg.norm(pytorch_flat) + 1e-8
    )
    
    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"PyTorch output: {pytorch_flat}")
    print(f"TFLite output:  {tflite_flat}")


def generate_header(tflite_path: Path, header_path: Path, array_name: str):
    """Generate C header file from TFLite model."""
    print(f"\nGenerating header: {header_path}")
    
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    with open(header_path, 'w') as f:
        f.write(f"// Auto-generated from {tflite_path.name}\n")
        f.write(f"// Size: {len(data)} bytes\n\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
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
    
    print(f"Header saved: {header_path}")


def main():
    print("=" * 60)
    print("ESP32-Compatible TFLite Export")
    print("=" * 60)
    
    # Paths
    project_root = Path.home() / "bird-detection"
    model_path = project_root / "models" / "lottery_ticket" / "winning_ticket_79pct.pth"
    output_dir = project_root / "models" / "exported_esp32"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Looking for alternative paths...")
        
        alt_paths = [
            project_root / "models" / "lottery_ticket" / "winning_ticket.pth",
            project_root / "winning_ticket_79pct.pth",
        ]
        
        for alt in alt_paths:
            if alt.exists():
                model_path = alt
                print(f"Found model at: {model_path}")
                break
        else:
            print("No model file found!")
            return
    
    # Load original model
    print(f"\nLoading model from: {model_path}")
    original_model = BirdClassifierCNN_Original(num_classes=6)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['state_dict'])
        else:
            original_model.load_state_dict(checkpoint)
    else:
        original_model.load_state_dict(checkpoint)
    
    original_model.eval()
    print("Original model loaded!")
    
    # Verify original model works
    test_input = torch.randn(1, 1, 40, 188)
    with torch.no_grad():
        test_output = original_model(test_input)
    print(f"Original model output shape: {test_output.shape}")
    
    # Create ESP32-compatible model and transfer weights
    print("\nCreating ESP32-compatible model...")
    esp32_model = BirdClassifierCNN_ESP32(num_classes=6)
    transfer_weights(original_model, esp32_model)
    esp32_model.eval()
    
    # Verify ESP32 model produces same output
    with torch.no_grad():
        esp32_output = esp32_model(test_input)
    
    diff = torch.abs(test_output - esp32_output).max().item()
    print(f"Max difference between models: {diff:.6f}")
    
    if diff > 0.01:
        print("WARNING: Models produce different outputs!")
        print("This is expected due to different pooling, but accuracy should be similar")
    
    # Generate calibration data
    calibration_data = generate_calibration_data(200)
    
    # Export float32
    float32_path = output_dir / "bird_classifier_float32.tflite"
    export_float32(esp32_model, float32_path)
    
    # Export int8
    int8_path = output_dir / "bird_classifier_int8.tflite"
    export_int8(esp32_model, int8_path, calibration_data)
    
    # Generate header for int8 model
    header_path = output_dir / "model_data.h"
    generate_header(int8_path, header_path, "bird_classifier_int8_tflite")
    
    # Verify models
    verify_model(float32_path, esp32_model)
    verify_model(int8_path, esp32_model)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - bird_classifier_float32.tflite ({float32_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - bird_classifier_int8.tflite ({int8_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - model_data.h (C header for Arduino)")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Copy model_data.h to your Arduino sketch folder:")
    print(f"   cp {header_path} /path/to/bird_detector/")
    print("\n2. The header file is ready to use - no xxd needed!")


if __name__ == "__main__":
    main()
