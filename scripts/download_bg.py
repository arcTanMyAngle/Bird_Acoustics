#!/usr/bin/env python3
"""
download_bg.py - Download and curate background noise for bird classifier

Creates a robust "background" class that teaches the model to recognize
non-bird audio (silence, wind, traffic, rain, ambient) rather than
misfiring into bird classes.

Sources:
1. ESC-50 (environmental sounds)
2. UrbanSound8K (urban ambient) - optional, requires manual download
3. Field recordings placeholder (user-supplied)

Usage:
    uv run python scripts/download_bg.py
    uv run python scripts/download_bg.py --target 150 --include-urban
"""

import os
import argparse
import requests
import zipfile
import shutil
import csv
import random
from pathlib import Path
from tqdm import tqdm
import hashlib

# ==========================================
# CONFIGURATION
# ==========================================
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
BASE_DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/raw/background")
TEMP_DIR = BASE_DATA_DIR / "temp"

# Target composition for background class
# Total should equal target_samples (default 100)
DEFAULT_TARGET = 100

# ESC-50 categories to use (avoiding animal sounds that could confuse model)
ESC50_CATEGORIES = {
    # Environmental
    "rain": {"weight": 0.12, "description": "Rain sounds"},
    "wind": {"weight": 0.12, "description": "Wind sounds"},
    "thunderstorm": {"weight": 0.05, "description": "Thunder/storm"},
    "sea_waves": {"weight": 0.05, "description": "Ocean waves"},
    "crackling_fire": {"weight": 0.03, "description": "Fire crackling"},
    
    # Urban/Traffic
    "engine": {"weight": 0.10, "description": "Engine idling"},
    "car_horn": {"weight": 0.05, "description": "Car horns"},
    "siren": {"weight": 0.05, "description": "Sirens"},
    "train": {"weight": 0.05, "description": "Train sounds"},
    "airplane": {"weight": 0.05, "description": "Airplane overhead"},
    "helicopter": {"weight": 0.03, "description": "Helicopter"},
    
    # Indoor/Mechanical
    "clock_tick": {"weight": 0.05, "description": "Clock ticking (near-silence)"},
    "vacuum_cleaner": {"weight": 0.05, "description": "Vacuum cleaner"},
    "washing_machine": {"weight": 0.05, "description": "Washing machine"},
    "keyboard_typing": {"weight": 0.05, "description": "Keyboard typing"},
    
    # Human activity (non-speech)
    "footsteps": {"weight": 0.05, "description": "Footsteps"},
    "door_wood_knock": {"weight": 0.05, "description": "Door knocking"},
}

# Categories to EXCLUDE (could confuse bird classifier)
ESC50_EXCLUDED = {
    # Animal sounds - explicit exclusion
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
    "sheep", "crow",  # Explicitly exclude crow sounds from ESC-50
    
    # Human speech/voice
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "laughing", "snoring",
    
    # Music/instruments
    "church_bells", "glass_breaking",
}


def ensure_dir(directory: Path):
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, filepath: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        
        with open(filepath, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_esc50(temp_dir: Path) -> Path | None:
    """Download and extract ESC-50 dataset."""
    zip_path = temp_dir / "esc50.zip"
    extract_path = temp_dir / "esc50"
    
    ensure_dir(temp_dir)
    
    # Download if not exists
    if not zip_path.exists():
        print("Downloading ESC-50 dataset (~600 MB)...")
        if not download_with_progress(ESC50_URL, zip_path, "ESC-50"):
            return None
    else:
        print("ESC-50 ZIP already exists, skipping download.")
    
    # Extract
    if not extract_path.exists():
        print("Extracting ESC-50...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_path)
        except zipfile.BadZipFile:
            print("Error: Corrupt ZIP file. Delete and retry.")
            return None
    
    # Find extracted folder (usually ESC-50-master)
    extracted_dirs = list(extract_path.glob("ESC-50*"))
    if not extracted_dirs:
        print("Error: Could not find extracted ESC-50 folder")
        return None
    
    return extracted_dirs[0]


def select_esc50_samples(
    esc50_root: Path,
    target_total: int,
    seed: int = 42
) -> list[tuple[Path, str]]:
    """
    Select samples from ESC-50 based on category weights.
    
    Returns list of (source_path, category) tuples.
    """
    audio_dir = esc50_root / "audio"
    meta_path = esc50_root / "meta" / "esc50.csv"
    
    if not meta_path.exists():
        print(f"Error: Metadata not found at {meta_path}")
        return []
    
    # Load metadata
    category_files: dict[str, list[Path]] = {cat: [] for cat in ESC50_CATEGORIES}
    
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            if cat in ESC50_CATEGORIES:
                filepath = audio_dir / row["filename"]
                if filepath.exists():
                    category_files[cat].append(filepath)
    
    # Report available files
    print("\nESC-50 category availability:")
    for cat, files in category_files.items():
        print(f"  {cat}: {len(files)} files")
    
    # Calculate samples per category based on weights
    total_weight = sum(info["weight"] for info in ESC50_CATEGORIES.values())
    
    random.seed(seed)
    selected: list[tuple[Path, str]] = []
    
    for cat, info in ESC50_CATEGORIES.items():
        n_target = int(target_total * info["weight"] / total_weight)
        available = category_files[cat]
        
        if len(available) == 0:
            print(f"  Warning: No files for {cat}")
            continue
        
        # Sample with replacement if needed (shouldn't happen for ESC-50)
        n_select = min(n_target, len(available))
        chosen = random.sample(available, n_select)
        
        for filepath in chosen:
            selected.append((filepath, cat))
    
    # If we're short, fill from largest categories
    while len(selected) < target_total:
        # Find category with most remaining files
        remaining = {
            cat: [f for f in files if (f, cat) not in selected]
            for cat, files in category_files.items()
        }
        best_cat = max(remaining.keys(), key=lambda c: len(remaining[c]))
        
        if not remaining[best_cat]:
            break
        
        extra = remaining[best_cat][0]
        selected.append((extra, best_cat))
    
    print(f"\nSelected {len(selected)} samples from ESC-50")
    return selected


def create_silence_samples(output_dir: Path, n_samples: int = 5, duration: float = 5.0):
    """
    Create synthetic silence/low-noise samples.
    
    These represent "nothing happening" - just recorder noise.
    """
    import numpy as np
    import soundfile as sf
    
    sample_rate = 16000
    n_samples_audio = int(sample_rate * duration)
    
    print(f"Creating {n_samples} synthetic silence samples...")
    
    for i in range(n_samples):
        # Very low amplitude noise (simulates recorder self-noise)
        noise_level = random.uniform(0.001, 0.005)
        audio = np.random.randn(n_samples_audio) * noise_level
        
        # Add slight low-frequency rumble (simulates ambient)
        t = np.linspace(0, duration, n_samples_audio)
        rumble = 0.002 * np.sin(2 * np.pi * random.uniform(20, 60) * t)
        audio = audio + rumble
        
        # Normalize to prevent clipping
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.01
        
        output_path = output_dir / f"silence_synthetic_{i:03d}.wav"
        sf.write(output_path, audio.astype(np.float32), sample_rate)
    
    print(f"  Created {n_samples} silence samples")


def copy_and_rename_samples(
    samples: list[tuple[Path, str]],
    output_dir: Path,
    prefix: str = "bg"
) -> int:
    """Copy selected samples to output directory with consistent naming."""
    ensure_dir(output_dir)
    
    copied = 0
    for i, (src_path, category) in enumerate(tqdm(samples, desc="Copying samples")):
        # Create filename: bg_category_index.wav
        ext = src_path.suffix
        dst_name = f"{prefix}_{category}_{i:04d}{ext}"
        dst_path = output_dir / dst_name
        
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
    
    return copied


def create_manifest(output_dir: Path):
    """Create a manifest file documenting the background samples."""
    manifest_path = output_dir / "MANIFEST.txt"
    
    files = sorted(output_dir.glob("*.wav")) + sorted(output_dir.glob("*.ogg"))
    
    # Count by category
    categories: dict[str, int] = {}
    for f in files:
        # Extract category from filename (bg_category_index.ext)
        parts = f.stem.split("_")
        if len(parts) >= 2:
            cat = parts[1]
            categories[cat] = categories.get(cat, 0) + 1
    
    with open(manifest_path, "w") as f:
        f.write("Background Noise Dataset Manifest\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files: {len(files)}\n\n")
        f.write("Category breakdown:\n")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            f.write(f"  {cat}: {count}\n")
        f.write("\n")
        f.write("Sources:\n")
        f.write("  - ESC-50 (environmental sounds)\n")
        f.write("  - Synthetic silence\n")
        f.write("\n")
        f.write("Excluded categories (to avoid confusion):\n")
        for cat in sorted(ESC50_EXCLUDED):
            f.write(f"  - {cat}\n")
    
    print(f"Created manifest: {manifest_path}")


def add_field_recordings_placeholder(output_dir: Path):
    """Create placeholder and instructions for user field recordings."""
    readme_path = output_dir / "FIELD_RECORDINGS_README.txt"
    
    content = """
Field Recordings for Background Class
======================================

For best real-world performance, add your own field recordings from your
actual deployment environment. These should capture:

1. SILENCE (10-20 samples)
   - Early morning quiet
   - Night ambient (no birds)
   - Indoor quiet

2. WIND (10-15 samples)
   - Light breeze
   - Moderate wind
   - Gusty conditions

3. TRAFFIC (10-15 samples)
   - Distant highway
   - Occasional car pass-by
   - Urban street noise

4. RAIN (5-10 samples)
   - Light drizzle
   - Steady rain
   - Rain on leaves/roof

5. AMBIENT NATURE (10-15 samples)
   - Rustling leaves
   - Insects (cicadas, crickets)
   - Water (creek, fountain)

Recording tips:
- Use same hardware (XIAO ESP32S3 Sense) if possible
- Record 30-60 second clips, will be split into 3s windows
- Capture at different times of day
- Include edge cases (loud traffic, heavy wind)

File naming convention:
  field_[category]_[index].wav
  Examples:
    field_wind_001.wav
    field_traffic_001.wav
    field_silence_001.wav

After adding recordings, re-run preprocessing:
  uv run python scripts/preprocess_audio.py
  uv run python scripts/augment_audio.py
"""
    
    with open(readme_path, "w") as f:
        f.write(content.strip())
    
    print(f"Created field recordings guide: {readme_path}")


def cleanup_temp_files(temp_dir: Path, keep_zip: bool = False):
    """Clean up temporary download files."""
    if temp_dir.exists():
        # Remove extracted folder
        for item in temp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif not keep_zip or item.suffix != ".zip":
                item.unlink()
        
        # Remove temp dir if empty
        if not any(temp_dir.iterdir()):
            temp_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(
        description="Download and curate background noise dataset"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Target number of samples (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded ZIP files for future use",
    )
    parser.add_argument(
        "--silence-samples",
        type=int,
        default=5,
        help="Number of synthetic silence samples to create",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    temp_dir = TEMP_DIR
    
    print("=" * 60)
    print("Background Noise Dataset Builder")
    print("=" * 60)
    print(f"Target samples: {args.target}")
    print(f"Output: {output_dir.absolute()}")
    print("=" * 60)
    
    # Check existing files
    ensure_dir(output_dir)
    existing = list(output_dir.glob("*.wav")) + list(output_dir.glob("*.ogg"))
    if existing:
        print(f"\nFound {len(existing)} existing files in {output_dir}")
        response = input("Clear and rebuild? [y/N]: ").strip().lower()
        if response == "y":
            for f in existing:
                f.unlink()
            print("Cleared existing files.")
        else:
            print("Keeping existing files. Exiting.")
            return
    
    # Step 1: Download ESC-50
    print("\n[1/4] Downloading ESC-50...")
    esc50_root = download_esc50(temp_dir)
    
    if esc50_root is None:
        print("Failed to download ESC-50. Exiting.")
        return
    
    # Step 2: Select samples
    print("\n[2/4] Selecting samples...")
    # Reserve some slots for silence
    esc50_target = args.target - args.silence_samples
    samples = select_esc50_samples(esc50_root, esc50_target, args.seed)
    
    # Step 3: Copy samples
    print("\n[3/4] Copying samples...")
    copied = copy_and_rename_samples(samples, output_dir)
    
    # Step 4: Create silence samples
    print("\n[4/4] Creating silence samples...")
    try:
        create_silence_samples(output_dir, args.silence_samples)
    except ImportError:
        print("  Skipping silence generation (numpy/soundfile not available)")
        print("  Run: uv pip install numpy soundfile")
    
    # Create manifest and instructions
    create_manifest(output_dir)
    add_field_recordings_placeholder(output_dir)
    
    # Cleanup
    if not args.keep_downloads:
        print("\nCleaning up temporary files...")
        cleanup_temp_files(temp_dir, keep_zip=False)
    
    # Summary
    final_count = len(list(output_dir.glob("*.wav")) + list(output_dir.glob("*.ogg")))
    
    print("\n" + "=" * 60)
    print("Background Dataset Ready!")
    print("=" * 60)
    print(f"Total samples: {final_count}")
    print(f"Location: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. (Optional) Add field recordings - see FIELD_RECORDINGS_README.txt")
    print("  2. Run preprocessing: uv run python scripts/preprocess_audio.py")
    print("  3. Run augmentation: uv run python scripts/augment_audio.py")


if __name__ == "__main__":
    main()