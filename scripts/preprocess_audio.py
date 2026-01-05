# ~/bird-detection/scripts/preprocess_audio.py

import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
SAMPLE_RATE = 16000  # 16kHz - matches XIAO's PDM mic
DURATION = 3.0  # 3-second clips
SAMPLES_PER_FILE = 3  # Extract up to 3 clips per recording


def process_audio_file(input_path, output_dir, class_name, file_idx):
    """Process a single audio file into fixed-length clips."""
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        # Skip if too short
        min_samples = int(SAMPLE_RATE * DURATION)
        if len(y) < min_samples:
            return 0
        
        clips_saved = 0
        
        # Extract multiple clips from longer recordings
        total_samples = len(y)
        clip_samples = int(SAMPLE_RATE * DURATION)
        
        # Calculate how many clips we can extract
        possible_clips = total_samples // clip_samples
        num_clips = min(possible_clips, SAMPLES_PER_FILE)
        
        for i in range(num_clips):
            start = i * clip_samples
            end = start + clip_samples
            clip = y[start:end]
            
            # Normalize
            clip = clip / (np.max(np.abs(clip)) + 1e-8)
            
            # Save
            output_path = output_dir / f"{class_name}_{file_idx:04d}_{i}.wav"
            sf.write(output_path, clip, SAMPLE_RATE)
            clips_saved += 1
        
        return clips_saved
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0


def process_class(class_name):
    """Process all files for a single class."""
    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = list(input_class_dir.glob("*.mp3")) + \
                  list(input_class_dir.glob("*.wav")) + \
                  list(input_class_dir.glob("*.ogg"))
    
    total_clips = 0
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc=class_name)):
        clips = process_audio_file(audio_file, output_class_dir, class_name, idx)
        total_clips += clips
    
    return total_clips


def main():
    print("=" * 60)
    print("Audio Preprocessing Pipeline")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Clip duration: {DURATION} seconds")
    print(f"Max clips per file: {SAMPLES_PER_FILE}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    summary = {}
    for class_name in classes:
        clips = process_class(class_name)
        summary[class_name] = clips
    
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    for class_name, count in summary.items():
        print(f"  {class_name}: {count} clips")
    print(f"\n  Total: {sum(summary.values())} clips")


if __name__ == "__main__":
    main()
