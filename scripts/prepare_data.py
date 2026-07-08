#!/usr/bin/env python3
"""
prepare_data.py - Master orchestration script for data pipeline

Runs the complete data preparation pipeline:
1. Download bird audio from Xeno-canto
2. Download background noise from ESC-50
3. Preprocess all audio (resample, clip, normalize)
4. Augment training data (noise injection, SpecAugment)
5. Validate dataset integrity

Usage:
    # Full pipeline (recommended first run)
    uv run python scripts/prepare_data.py --full
    
    # Download only (if preprocessing already done)
    uv run python scripts/prepare_data.py --download-only
    
    # Skip downloads (use existing raw data)
    uv run python scripts/prepare_data.py --skip-download
    
    # Validate existing dataset
    uv run python scripts/prepare_data.py --validate-only
"""

import argparse
import subprocess
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"

# Expected classes (alphabetically sorted for consistent indexing)
EXPECTED_CLASSES = [
    "american_crow",
    "background",
    "california_quail",
    "california_scrub_jay",
    "great_horned_owl",
    "killdeer",
    "mourning_dove",
    "red_tailed_hawk",
    "western_meadowlark",
]

# Minimum samples per class
MIN_SAMPLES_RAW = 50
MIN_SAMPLES_PROCESSED = 100
MIN_SAMPLES_AUGMENTED = 200


def run_script(script_name: str, args: list[str] = None, cwd: Path = None) -> bool:
    """Run a Python script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False
    
    cmd = ["uv", "run", "python", str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def count_samples(directory: Path) -> dict[str, int]:
    """Count audio samples per class in a directory."""
    counts = {}
    
    if not directory.exists():
        return counts
    
    for class_dir in directory.iterdir():
        if class_dir.is_dir():
            # Count audio files
            audio_files = (
                list(class_dir.glob("*.wav")) +
                list(class_dir.glob("*.mp3")) +
                list(class_dir.glob("*.ogg"))
            )
            counts[class_dir.name] = len(audio_files)
    
    return counts


def validate_class_structure(directory: Path, min_samples: int, stage: str) -> bool:
    """Validate that all expected classes exist with minimum samples."""
    print(f"\nValidating {stage} data in {directory}...")
    
    if not directory.exists():
        print(f"  Error: Directory does not exist: {directory}")
        return False
    
    counts = count_samples(directory)
    
    # Check for missing classes
    missing = set(EXPECTED_CLASSES) - set(counts.keys())
    if missing:
        print(f"  Warning: Missing classes: {missing}")
    
    # Check for unexpected classes
    unexpected = set(counts.keys()) - set(EXPECTED_CLASSES)
    if unexpected:
        print(f"  Note: Additional classes found: {unexpected}")
    
    # Check sample counts
    all_ok = True
    for cls in EXPECTED_CLASSES:
        count = counts.get(cls, 0)
        status = "✓" if count >= min_samples else "✗"
        print(f"  {status} {cls}: {count} samples (min: {min_samples})")
        if count < min_samples:
            all_ok = False
    
    return all_ok


def compute_dataset_hash(directory: Path) -> str:
    """Compute a hash of the dataset for reproducibility tracking."""
    if not directory.exists():
        return "N/A"
    
    # Hash based on file count and total size
    total_files = 0
    total_size = 0
    
    for audio_file in directory.rglob("*.wav"):
        total_files += 1
        total_size += audio_file.stat().st_size
    
    for audio_file in directory.rglob("*.mp3"):
        total_files += 1
        total_size += audio_file.stat().st_size
    
    hash_input = f"{total_files}:{total_size}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def save_manifest(output_path: Path):
    """Save a manifest file documenting the dataset."""
    manifest = {
        "created": datetime.now().isoformat(),
        "classes": EXPECTED_CLASSES,
        "stages": {},
    }
    
    for stage, directory in [
        ("raw", RAW_DIR),
        ("processed", PROCESSED_DIR),
        ("augmented", AUGMENTED_DIR),
    ]:
        counts = count_samples(directory)
        manifest["stages"][stage] = {
            "directory": str(directory),
            "samples_per_class": counts,
            "total_samples": sum(counts.values()),
            "hash": compute_dataset_hash(directory),
        }
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSaved manifest: {output_path}")


def print_summary():
    """Print a summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for stage, directory, min_samples in [
        ("Raw", RAW_DIR, MIN_SAMPLES_RAW),
        ("Processed", PROCESSED_DIR, MIN_SAMPLES_PROCESSED),
        ("Augmented", AUGMENTED_DIR, MIN_SAMPLES_AUGMENTED),
    ]:
        counts = count_samples(directory)
        total = sum(counts.values())
        
        print(f"\n{stage} ({directory}):")
        print(f"  Total: {total} samples across {len(counts)} classes")
        
        if counts:
            min_count = min(counts.values())
            max_count = max(counts.values())
            print(f"  Range: {min_count} - {max_count} per class")
            
            # Check balance
            if max_count > 0:
                balance = min_count / max_count
                balance_status = "✓ Good" if balance > 0.5 else "⚠ Imbalanced"
                print(f"  Balance: {balance:.2f} ({balance_status})")


def download_bird_audio(target: int = 100):
    """Download bird audio from Xeno-canto."""
    return run_script(
        "download_xenocanto.py",
        ["--recommended-only", "--target", str(target)]
    )


def download_background_audio(target: int = 100):
    """Download background noise from ESC-50."""
    return run_script(
        "download_bg.py",
        ["--target", str(target)]
    )


def preprocess_audio():
    """Preprocess all raw audio."""
    return run_script("preprocess_audio.py")


def augment_audio():
    """Augment processed audio."""
    return run_script("augment_audio.py")


def main():
    parser = argparse.ArgumentParser(
        description="Master data preparation pipeline"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (download + preprocess + augment)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw data",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads, process existing raw data",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=100,
        help="Target samples per class for downloads (default: 100)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean processed/augmented dirs before running",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("California Bird Acoustic Detection - Data Pipeline")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Expected classes: {len(EXPECTED_CLASSES)}")
    print(f"Target samples: {args.target_samples}")
    
    # Validate only
    if args.validate_only:
        validate_class_structure(RAW_DIR, MIN_SAMPLES_RAW, "raw")
        validate_class_structure(PROCESSED_DIR, MIN_SAMPLES_PROCESSED, "processed")
        validate_class_structure(AUGMENTED_DIR, MIN_SAMPLES_AUGMENTED, "augmented")
        print_summary()
        return
    
    # Clean if requested
    if args.clean:
        print("\nCleaning processed and augmented directories...")
        for directory in [PROCESSED_DIR, AUGMENTED_DIR]:
            if directory.exists():
                shutil.rmtree(directory)
                print(f"  Removed: {directory}")
    
    # Download phase
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("PHASE 1: DOWNLOADING DATA")
        print("=" * 60)
        
        if not download_bird_audio(args.target_samples):
            print("Warning: Bird audio download had issues")
        
        if not download_background_audio(args.target_samples):
            print("Warning: Background audio download had issues")
        
        if args.download_only:
            validate_class_structure(RAW_DIR, MIN_SAMPLES_RAW, "raw")
            print("\nDownload complete. Run with --skip-download to process.")
            return
    
    # Preprocess phase
    print("\n" + "=" * 60)
    print("PHASE 2: PREPROCESSING")
    print("=" * 60)
    
    if not preprocess_audio():
        print("Error: Preprocessing failed")
        return
    
    # Validate preprocessed
    if not validate_class_structure(PROCESSED_DIR, MIN_SAMPLES_PROCESSED // 2, "processed"):
        print("Warning: Preprocessed data may be incomplete")
    
    # Augment phase
    print("\n" + "=" * 60)
    print("PHASE 3: AUGMENTATION")
    print("=" * 60)
    
    if not augment_audio():
        print("Error: Augmentation failed")
        return
    
    # Final validation
    print("\n" + "=" * 60)
    print("PHASE 4: VALIDATION")
    print("=" * 60)
    
    validate_class_structure(AUGMENTED_DIR, MIN_SAMPLES_AUGMENTED // 2, "augmented")
    
    # Save manifest
    manifest_path = DATA_DIR / "dataset_manifest.json"
    save_manifest(manifest_path)
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Review manifest: {manifest_path}")
    print(f"  2. Train model: uv run python scripts/train_v3.py --data-dir {AUGMENTED_DIR}")
    print(f"  3. Export model: uv run python scripts/export_v2.py --model-path models/v3/best_model.pth")


if __name__ == "__main__":
    main()