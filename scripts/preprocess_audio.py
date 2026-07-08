#!/usr/bin/env python3
"""
preprocess_audio.py - Energy-filtered audio preprocessing

Extracts 3-second clips that contain actual vocalizations,
not silence. Critical for sparse callers like owls.

Background class is exempted from filtering (we want all background samples).
"""

from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
SAMPLE_RATE = 16000
DURATION = 3.0
MAX_CLIPS_PER_FILE = 5

# Energy filtering parameters
MIN_ACTIVE_RATIO = 0.4
RELATIVE_PERCENTILE = 70
MIN_DB_ABOVE_FLOOR = 10.0

# Species-specific frequency bands (Hz)
SPECIES_FREQ_BANDS = {
    'great_horned_owl': (150, 1200),
    'mourning_dove': (300, 900),
    'american_crow': (500, 3500),
    'california_quail': (1500, 5000),
    'california_scrub_jay': (1000, 5000),
    'killdeer': (2000, 6000),
    'red_tailed_hawk': (2000, 4500),
    'western_meadowlark': (1500, 6000),
    'background': (200, 8000),
}
DEFAULT_FREQ_BAND = (200, 8000)

# Classes to exempt from energy filtering
SKIP_FILTERING_CLASSES = {'background'}


def compute_band_energy(
    audio: np.ndarray,
    sr: int,
    freq_low: int,
    freq_high: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Compute energy in target frequency band per frame."""
    D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    magnitude = np.abs(D[freq_mask, :])
    energy = np.sum(magnitude ** 2, axis=0)
    
    return energy


def find_active_windows(
    audio: np.ndarray,
    sr: int,
    freq_band: tuple[int, int],
    window_sec: float = 3.0,
    hop_sec: float = 0.5,
    min_active_ratio: float = 0.4,
    relative_percentile: float = 70,
    min_db_above_floor: float = 10.0
) -> list[tuple[int, int, float]]:
    """
    Find windows containing actual vocalizations using hybrid threshold.
    
    Uses the HIGHER of:
    1. Relative threshold (70th percentile of recording)
    2. Absolute threshold (10dB above noise floor)
    
    Returns list of (start_sample, end_sample, energy_score) sorted by energy.
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    freq_low, freq_high = freq_band
    energy = compute_band_energy(audio, sr, freq_low, freq_high)
    
    if len(energy) == 0:
        return []
    
    # Convert to dB
    energy_db = 10 * np.log10(energy + 1e-10)
    
    # Two thresholds:
    # 1. Relative: Xth percentile of this recording
    relative_thresh = np.percentile(energy_db, relative_percentile)
    
    # 2. Absolute: must be X dB above the noise floor (10th percentile)
    noise_floor = np.percentile(energy_db, 10)
    absolute_thresh = noise_floor + min_db_above_floor
    
    # Use the HIGHER of the two thresholds
    threshold = max(relative_thresh, absolute_thresh)
    
    candidates = []
    hop_frames = 512
    
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        
        frame_start = start // hop_frames
        frame_end = min(end // hop_frames, len(energy_db))
        
        if frame_end <= frame_start:
            continue
        
        window_energy_db = energy_db[frame_start:frame_end]
        active_frames = np.sum(window_energy_db > threshold)
        active_ratio = active_frames / len(window_energy_db)
        
        if active_ratio >= min_active_ratio:
            mean_energy = np.mean(window_energy_db)
            candidates.append((start, end, mean_energy))
    
    # Sort by energy (highest first), remove overlaps
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    for start, end, score in candidates:
        overlaps = any(not (end <= s or start >= e) for s, e, _ in selected)
        if not overlaps:
            selected.append((start, end, score))
    
    return selected


def process_audio_file_filtered(
    input_path: Path,
    output_dir: Path,
    class_name: str,
    file_idx: int,
    freq_band: tuple[int, int]
) -> int:
    """Process a single audio file with energy filtering."""
    try:
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        min_samples = int(SAMPLE_RATE * DURATION)
        if len(y) < min_samples:
            return 0
        
        # Find active windows
        windows = find_active_windows(
            y, sr, freq_band,
            window_sec=DURATION,
            hop_sec=0.5,
            min_active_ratio=MIN_ACTIVE_RATIO,
            relative_percentile=RELATIVE_PERCENTILE,
            min_db_above_floor=MIN_DB_ABOVE_FLOOR
        )
        
        # Limit clips per file
        windows = windows[:MAX_CLIPS_PER_FILE]
        
        clips_saved = 0
        for i, (start, end, _) in enumerate(windows):
            clip = y[start:end]
            
            # Normalize
            max_val = np.max(np.abs(clip))
            if max_val > 1e-8:
                clip = clip / max_val
            
            stem = input_path.stem
            output_path = output_dir / f"{class_name}_{file_idx:04d}_{stem}_w{i}.wav"
            sf.write(output_path, clip, SAMPLE_RATE)
            clips_saved += 1
        
        return clips_saved
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0


def process_audio_file_simple(
    input_path: Path,
    output_dir: Path,
    class_name: str,
    file_idx: int
) -> int:
    """Process without energy filtering (for background class)."""
    try:
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        min_samples = int(SAMPLE_RATE * DURATION)
        if len(y) < min_samples:
            return 0
        
        clips_saved = 0
        clip_samples = int(SAMPLE_RATE * DURATION)
        # Overlapping windows (1 s hop): 5 s ESC-50 clips yield 3 windows instead of 1.
        # Same source file = same {class}_{idx} group, so overlap cannot leak across splits.
        hop_samples = SAMPLE_RATE
        starts = list(range(0, len(y) - clip_samples + 1, hop_samples))[:MAX_CLIPS_PER_FILE]

        for i, start in enumerate(starts):
            clip = y[start:start + clip_samples]
            
            max_val = np.max(np.abs(clip))
            if max_val > 1e-8:
                clip = clip / max_val
            
            stem = input_path.stem
            output_path = output_dir / f"{class_name}_{file_idx:04d}_{stem}_w{i}.wav"
            sf.write(output_path, clip, SAMPLE_RATE)
            clips_saved += 1
        
        return clips_saved
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0


def process_class(class_name: str) -> dict:
    """Process all files for a single class."""
    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    freq_band = SPECIES_FREQ_BANDS.get(class_name, DEFAULT_FREQ_BAND)
    skip_filtering = class_name in SKIP_FILTERING_CLASSES
    
    audio_files = sorted(
        list(input_class_dir.glob("*.mp3")) +
        list(input_class_dir.glob("*.wav")) +
        list(input_class_dir.glob("*.ogg"))
    )
    
    total_clips = 0
    files_with_clips = 0
    files_skipped = 0
    
    desc = f"{class_name} (no filter)" if skip_filtering else f"{class_name} ({freq_band[0]}-{freq_band[1]}Hz)"
    
    for idx, audio_file in enumerate(tqdm(audio_files, desc=desc)):
        if skip_filtering:
            clips = process_audio_file_simple(audio_file, output_class_dir, class_name, idx)
        else:
            clips = process_audio_file_filtered(audio_file, output_class_dir, class_name, idx, freq_band)
        
        total_clips += clips
        if clips > 0:
            files_with_clips += 1
        else:
            files_skipped += 1
    
    return {
        'clips': total_clips,
        'files_with_clips': files_with_clips,
        'files_skipped': files_skipped,
        'total_files': len(audio_files),
        'retention_rate': files_with_clips / max(len(audio_files), 1) * 100
    }


def main():
    print("=" * 60)
    print("Energy-Filtered Audio Preprocessing")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Clip duration: {DURATION} seconds")
    print(f"Min active ratio: {MIN_ACTIVE_RATIO}")
    print(f"Relative percentile: {RELATIVE_PERCENTILE}")
    print(f"Min dB above floor: {MIN_DB_ABOVE_FLOOR}")
    print(f"Skip filtering for: {SKIP_FILTERING_CLASSES}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear existing processed data
    for existing in OUTPUT_DIR.rglob("*.wav"):
        existing.unlink()
    
    classes = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    
    print(f"\nProcessing {len(classes)} classes...\n")
    
    summary = {}
    for class_name in classes:
        stats = process_class(class_name)
        summary[class_name] = stats
    
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"{'Class':<25} {'Clips':>8} {'Files':>8} {'Retention':>10}")
    print("-" * 60)
    
    for class_name, stats in summary.items():
        print(f"{class_name:<25} {stats['clips']:>8} "
              f"{stats['files_with_clips']:>5}/{stats['total_files']:<3} "
              f"{stats['retention_rate']:>9.1f}%")
    
    total_clips = sum(s['clips'] for s in summary.values())
    print("-" * 60)
    print(f"{'Total':<25} {total_clips:>8}")
    
    # Flag species with low retention (excluding background-exempt classes)
    low_retention = [
        c for c, s in summary.items() 
        if s['retention_rate'] < 50 and c not in SKIP_FILTERING_CLASSES
    ]
    if low_retention:
        print(f"\n⚠️  Low retention species (may need more raw data):")
        for c in low_retention:
            print(f"   - {c}: {summary[c]['retention_rate']:.1f}%")


if __name__ == "__main__":
    main()