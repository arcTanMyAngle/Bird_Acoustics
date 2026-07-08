#!/usr/bin/env python3
"""
augment_audio.py - Data augmentation pipeline with noise mixing

Key features:
1. Standard augmentations (time stretch, pitch shift, gain, noise)
2. Background noise mixing (SNR-controlled) for robustness
3. Class-aware augmentation (more augments for underrepresented classes)
4. Reproducible with seed control

Usage:
    uv run python scripts/augment_audio.py
    uv run python scripts/augment_audio.py --augments-per-file 3 --mix-noise
"""

import os
import argparse
import random
from pathlib import Path
from typing import Optional
import shutil

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

try:
    from audiomentations import (
        Compose, AddGaussianNoise, TimeStretch, PitchShift,
        Shift, Gain, HighPassFilter, LowPassFilter
    )
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("Warning: audiomentations not installed. Using basic augmentations.")
    print("Install with: uv pip install audiomentations")


# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/augmented")
BACKGROUND_DIR = Path("data/processed/background")
SAMPLE_RATE = 16000

DEFAULT_AUGMENTS_PER_FILE = 2
DEFAULT_NOISE_MIX_PROB = 0.3  # Probability of mixing background noise
DEFAULT_SNR_RANGE = (5, 20)   # Signal-to-noise ratio range in dB


# ==========================================
# AUGMENTATION PIPELINES
# ==========================================

def create_augment_pipeline(seed: int = None):
    """Create audiomentations pipeline."""
    if not HAS_AUDIOMENTATIONS:
        return None
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    return Compose([
        # Additive noise
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        
        # Time/pitch modifications (more conservative for bird calls)
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        
        # Temporal shift
        Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
        
        # Gain variation
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        
        # Frequency filtering (simulates distance/environment)
        HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=500, p=0.2),
        LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=7500, p=0.2),
    ])


def basic_augment(y: np.ndarray, sr: int) -> np.ndarray:
    """Basic augmentation without audiomentations library."""
    aug_type = random.choice(["noise", "gain", "shift", "none"])
    
    if aug_type == "noise":
        noise_amp = random.uniform(0.001, 0.01)
        noise = np.random.randn(len(y)) * noise_amp
        y = y + noise
    
    elif aug_type == "gain":
        gain_db = random.uniform(-6, 6)
        y = y * (10 ** (gain_db / 20))
    
    elif aug_type == "shift":
        shift_samples = int(random.uniform(-0.2, 0.2) * len(y))
        y = np.roll(y, shift_samples)
    
    return y


# ==========================================
# NOISE MIXING
# ==========================================

class BackgroundNoiseMixer:
    """Mix background noise into bird audio at controlled SNR."""
    
    def __init__(
        self,
        background_dir: Path,
        sample_rate: int = 16000,
        snr_range: tuple[float, float] = (5, 20),
    ):
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.noise_files: list[Path] = []
        
        # Load list of background files
        if background_dir.exists():
            self.noise_files = sorted(background_dir.glob("*.wav"))
            print(f"Loaded {len(self.noise_files)} background noise files")
        else:
            print(f"Warning: Background directory not found: {background_dir}")
    
    def load_random_noise(self, target_length: int) -> Optional[np.ndarray]:
        """Load a random noise sample, repeating/trimming to target length."""
        if not self.noise_files:
            return None
        
        noise_file = random.choice(self.noise_files)
        
        try:
            noise, _ = librosa.load(noise_file, sr=self.sample_rate, mono=True)
            
            # Repeat if too short
            if len(noise) < target_length:
                repeats = (target_length // len(noise)) + 1
                noise = np.tile(noise, repeats)
            
            # Trim to exact length
            noise = noise[:target_length]
            
            return noise
        except Exception as e:
            print(f"Error loading noise {noise_file}: {e}")
            return None
    
    def mix(self, signal: np.ndarray, snr_db: float = None) -> np.ndarray:
        """Mix signal with background noise at specified SNR."""
        noise = self.load_random_noise(len(signal))
        
        if noise is None:
            return signal
        
        if snr_db is None:
            snr_db = random.uniform(*self.snr_range)
        
        # Calculate signal and noise power
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return signal
        
        # Calculate required noise scaling for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_scale = sqrt(signal_power / (noise_power * 10^(SNR/10)))
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise_scale = np.sqrt(target_noise_power / noise_power)
        
        # Mix
        mixed = signal + noise * noise_scale
        
        # Prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.99
        
        return mixed


# ==========================================
# MAIN AUGMENTATION LOGIC
# ==========================================

def augment_file(
    input_path: Path,
    output_dir: Path,
    base_name: str,
    aug_idx: int,
    augment_fn,
    noise_mixer: Optional[BackgroundNoiseMixer] = None,
    noise_mix_prob: float = 0.3,
    sample_rate: int = SAMPLE_RATE,
) -> bool:
    """Create an augmented version of an audio file."""
    try:
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        
        # Apply augmentation
        if augment_fn is not None:
            if HAS_AUDIOMENTATIONS:
                y_aug = augment_fn(samples=y, sample_rate=sr)
            else:
                y_aug = basic_augment(y, sr)
        else:
            y_aug = y.copy()
        
        # Optionally mix with background noise
        if noise_mixer is not None and random.random() < noise_mix_prob:
            y_aug = noise_mixer.mix(y_aug)
        
        # Normalize
        max_val = np.max(np.abs(y_aug))
        if max_val > 0:
            y_aug = y_aug / max_val * 0.95
        
        # Save
        output_path = output_dir / f"{base_name}_aug{aug_idx}.wav"
        sf.write(output_path, y_aug, sample_rate)
        
        return True
    except Exception as e:
        print(f"Error augmenting {input_path}: {e}")
        return False


def process_class(
    class_name: str,
    input_dir: Path,
    output_dir: Path,
    augments_per_file: int,
    augment_fn,
    noise_mixer: Optional[BackgroundNoiseMixer],
    noise_mix_prob: float,
    is_background: bool = False,
) -> tuple[int, int]:
    """Augment all files for a single class."""
    input_class_dir = input_dir / class_name
    output_class_dir = output_dir / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files (sorted: glob order is fs-dependent, would break seed reproducibility)
    audio_files = sorted(input_class_dir.glob("*.wav"))
    
    original_count = 0
    augmented_count = 0
    
    for audio_file in tqdm(audio_files, desc=f"{class_name}", leave=False):
        base_name = audio_file.stem
        
        # Copy original
        shutil.copy(audio_file, output_class_dir / audio_file.name)
        original_count += 1
        
        # Background gets full augmentation too: it is the FP-rejection class and has
        # the fewest source recordings, so it needs the variety more than the birds do.
        n_augs = augments_per_file
        
        # For background class, don't mix noise into noise
        mix_prob = 0.0 if is_background else noise_mix_prob
        
        # Create augmented versions
        for i in range(n_augs):
            if augment_file(
                audio_file,
                output_class_dir,
                base_name,
                i,
                augment_fn,
                noise_mixer,
                mix_prob,
            ):
                augmented_count += 1
    
    return original_count, augmented_count


def main():
    parser = argparse.ArgumentParser(description="Audio data augmentation pipeline")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(INPUT_DIR),
        help=f"Input directory (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--augments-per-file",
        type=int,
        default=DEFAULT_AUGMENTS_PER_FILE,
        help=f"Augmentations per file (default: {DEFAULT_AUGMENTS_PER_FILE})",
    )
    parser.add_argument(
        "--mix-noise",
        action="store_true",
        help="Mix background noise into bird audio (recommended)",
    )
    parser.add_argument(
        "--noise-mix-prob",
        type=float,
        default=DEFAULT_NOISE_MIX_PROB,
        help=f"Probability of mixing noise (default: {DEFAULT_NOISE_MIX_PROB})",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=DEFAULT_SNR_RANGE[0],
        help=f"Minimum SNR for noise mixing in dB (default: {DEFAULT_SNR_RANGE[0]})",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=DEFAULT_SNR_RANGE[1],
        help=f"Maximum SNR for noise mixing in dB (default: {DEFAULT_SNR_RANGE[1]})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Data Augmentation Pipeline")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per file: {args.augments_per_file}")
    print(f"Noise mixing: {'enabled' if args.mix_noise else 'disabled'}")
    if args.mix_noise:
        print(f"  Probability: {args.noise_mix_prob}")
        print(f"  SNR range: {args.snr_min} - {args.snr_max} dB")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create augmentation pipeline
    augment_fn = create_augment_pipeline(args.seed)
    
    # Create noise mixer
    noise_mixer = None
    if args.mix_noise:
        background_dir = input_dir / "background"
        if background_dir.exists():
            noise_mixer = BackgroundNoiseMixer(
                background_dir,
                snr_range=(args.snr_min, args.snr_max),
            )
        else:
            print(f"Warning: Background dir not found, disabling noise mixing")
    
    # Get all classes
    classes = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    
    total_original = 0
    total_augmented = 0
    
    print(f"\nProcessing {len(classes)} classes...")
    
    for class_name in classes:
        is_background = (class_name == "background")
        
        orig, aug = process_class(
            class_name,
            input_dir,
            output_dir,
            args.augments_per_file,
            augment_fn,
            noise_mixer,
            args.noise_mix_prob,
            is_background,
        )
        
        total_original += orig
        total_augmented += aug
        
        total_class = orig + aug
        print(f"  {class_name}: {orig} original + {aug} augmented = {total_class} total")
    
    print("\n" + "=" * 60)
    print("Augmentation Summary")
    print("=" * 60)
    print(f"  Original clips: {total_original}")
    print(f"  Augmented clips: {total_augmented}")
    print(f"  Total dataset: {total_original + total_augmented}")
    
    # Verify class balance
    print("\nClass distribution:")
    for class_name in classes:
        class_dir = output_dir / class_name
        count = len(list(class_dir.glob("*.wav")))
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()