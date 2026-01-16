#!/usr/bin/env python3
"""scripts/create_manifest.py - Data integrity tracking"""

import hashlib
import json
from pathlib import Path
from tqdm import tqdm

def md5_file(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def create_manifest(data_dir: Path, output_path: Path):
    """Create manifest of all processed audio files."""
    manifest = {
        "version": "3.0",
        "created": "2026-01-08",
        "files": {}
    }
    
    audio_files = list(data_dir.rglob("*.wav"))
    print(f"Hashing {len(audio_files)} files...")
    
    for filepath in tqdm(audio_files):
        rel_path = str(filepath.relative_to(data_dir))
        manifest["files"][rel_path] = {
            "md5": md5_file(filepath),
            "size": filepath.stat().st_size
        }
    
    # Summary by class
    manifest["summary"] = {}
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.wav")))
            manifest["summary"][class_dir.name] = count
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Manifest saved to {output_path}")
    print(f"   Total files: {len(manifest['files'])}")
    for cls, count in manifest["summary"].items():
        print(f"   {cls}: {count}")

if __name__ == "__main__":
    create_manifest(
        Path("data/processed"),
        Path("data/manifest.json")
    )