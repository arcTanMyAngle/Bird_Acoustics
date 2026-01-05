# ~/bird-detection/scripts/augment_audio.py

import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Gain
)

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/augmented")
SAMPLE_RATE = 16000
AUGMENTATIONS_PER_FILE = 2  # Create 2 augmented versions of each file


# Define augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
])


def augment_file(input_path, output_dir, base_name, aug_idx):
    """Create an augmented version of an audio file."""
    try:
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        # Apply augmentation
        y_aug = augment(samples=y, sample_rate=sr)
        
        # Normalize
        y_aug = y_aug / (np.max(np.abs(y_aug)) + 1e-8)
        
        # Save
        output_path = output_dir / f"{base_name}_aug{aug_idx}.wav"
        sf.write(output_path, y_aug, SAMPLE_RATE)
        
        return True
    except Exception as e:
        print(f"Error augmenting {input_path}: {e}")
        return False


def process_class(class_name):
    """Augment all files for a single class."""
    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    # First, copy original files
    audio_files = list(input_class_dir.glob("*.wav"))
    
    augmented_count = 0
    
    for audio_file in tqdm(audio_files, desc=f"{class_name}"):
        base_name = audio_file.stem
        
        # Copy original
        shutil.copy(audio_file, output_class_dir / audio_file.name)
        
        # Create augmented versions
        for i in range(AUGMENTATIONS_PER_FILE):
            if augment_file(audio_file, output_class_dir, base_name, i):
                augmented_count += 1
    
    original_count = len(audio_files)
    return original_count, augmented_count


def main():
    print("=" * 60)
    print("Data Augmentation Pipeline")
    print("=" * 60)
    print(f"Augmentations per file: {AUGMENTATIONS_PER_FILE}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    total_original = 0
    total_augmented = 0
    
    for class_name in classes:
        orig, aug = process_class(class_name)
        total_original += orig
        total_augmented += aug
        print(f"  {class_name}: {orig} original + {aug} augmented = {orig + aug} total")
    
    print("\n" + "=" * 60)
    print("Augmentation Summary")
    print("=" * 60)
    print(f"  Original clips: {total_original}")
    print(f"  Augmented clips: {total_augmented}")
    print(f"  Total dataset: {total_original + total_augmented}")


if __name__ == "__main__":
    main()
