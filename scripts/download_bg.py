import os
import requests
import zipfile
import shutil
import csv
from pathlib import Path
from tqdm import tqdm  # Added for progress bar since the file is large

# ==========================================
# CONFIGURATION
# ==========================================
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
BASE_DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/raw/background")
TEMP_ZIP = BASE_DATA_DIR / "esc50.zip"

# Categories to keep as "background" noise
# We exclude animals (dog, rooster, etc.) to avoid confusion, 
# keeping only environmental and urban sounds.
BACKGROUND_CATEGORIES = {
    "rain", "sea_waves", "crackling_fire", "thunderstorm", "wind", 
    "helicopter", "chainsaw", "engine", "train", "airplane", 
    "footsteps", "door_wood_knock", "keyboard_typing", "washing_machine", 
    "clock_tick", "vacuum_cleaner", "car_horn", "siren"
}

def ensure_dir(directory):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def download_esc50():
    print("=" * 60)
    print("ESC-50 Background Noise Downloader")
    print("=" * 60)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(BASE_DATA_DIR)

    # 1. Download the dataset
    if not TEMP_ZIP.exists():
        print("Downloading ESC-50 dataset (approx. 600 MB)...")
        try:
            response = requests.get(ESC50_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(TEMP_ZIP, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            print(" ✓ Download complete.")
        except Exception as e:
            print(f" X Error downloading: {e}")
            return
    else:
        print(" ! ZIP file already exists, skipping download.")

    # 2. Extract
    print("Extracting files...")
    extract_path = BASE_DATA_DIR
    try:
        with zipfile.ZipFile(TEMP_ZIP, "r") as z:
            z.extractall(extract_path)
    except zipfile.BadZipFile:
        print(" X Error: The zip file is corrupted. Delete 'data/esc50.zip' and try again.")
        return

    # Paths inside the extracted folder
    # GitHub archive usually extracts to "ESC-50-master"
    extracted_root = extract_path / "ESC-50-master"
    audio_dir = extracted_root / "audio"
    meta_path = extracted_root / "meta" / "esc50.csv"

    if not meta_path.exists():
        print(f" X Error: Could not find metadata at {meta_path}")
        return

    # 3. Filter and Copy
    print(f"Filtering for {len(BACKGROUND_CATEGORIES)} background categories...")
    
    copied_count = 0
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in BACKGROUND_CATEGORIES:
                src_file = audio_dir / row["filename"]
                dst_file = OUTPUT_DIR / row["filename"]
                
                # Copy if it exists
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1

    print(f" ✓ Copied {copied_count} background audio files to {OUTPUT_DIR}")

    # 4. Cleanup
    print("Cleaning up temporary files...")
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    
    # Optional: Keep the zip if you want to save bandwidth later, 
    # but the guide suggests cleaning it up.
    if TEMP_ZIP.exists():
        TEMP_ZIP.unlink()
    
    print("=" * 60)
    print("Background dataset ready!")

if __name__ == "__main__":
    download_esc50()
