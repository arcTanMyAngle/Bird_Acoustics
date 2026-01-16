#!/usr/bin/env python3
"""scripts/visualize_spectrograms.py - Level 2 Boss Fight"""

import random
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def visualize_random_spectrograms(data_dir: Path, n_samples: int = 3):
    """Generate visualization of random spectrograms."""
    print("=" * 60)
    print("🎮 LEVEL 2 BOSS FIGHT: Spectrogram Visualization")
    print("=" * 60)
    
    # Collect all audio files
    audio_files = list(data_dir.rglob("*.wav"))
    if len(audio_files) < n_samples:
        print(f"❌ Not enough audio files. Found {len(audio_files)}, need {n_samples}")
        return False
    
    # Random sample
    samples = random.sample(audio_files, n_samples)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
    
    for idx, audio_path in enumerate(samples):
        class_name = audio_path.parent.name
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Compute mel spectrogram (same params as training)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, n_fft=512, hop_length=256
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot waveform
        ax_wave = axes[idx, 0]
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set_title(f"Waveform: {class_name}", fontsize=12)
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        
        # Plot mel spectrogram
        ax_mel = axes[idx, 1]
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel', ax=ax_mel
        )
        ax_mel.set_title(f"Mel Spectrogram: {class_name}", fontsize=12)
        fig.colorbar(img, ax=ax_mel, format='%+2.0f dB')
    
    plt.tight_layout()
    output_path = Path("models/spectrogram_validation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved to {output_path}")
    print("\n🔍 MANUAL CHECK REQUIRED:")
    print("   Open the image and verify:")
    print("   1. Bird calls show clear harmonic patterns (horizontal bands)")
    print("   2. Background noise shows diffuse energy distribution")
    print("   3. No corrupted/silent files (all-black spectrograms)")
    print("\n" + "=" * 60)
    print("🏆 BOSS FIGHT CONDITION: Visual inspection confirms data quality")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    visualize_random_spectrograms(Path("data/processed"))