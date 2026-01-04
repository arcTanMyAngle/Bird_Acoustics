import os
import requests
import json
import time
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = ""
BASE_URL = "https://xeno-canto.org/api/3/recordings"
OUTPUT_DIR = Path("data/raw")
TARGET_SAMPLES = 100

# Species Mapping (Folder Name -> Scientific Name)
SPECIES = {
    "western_meadowlark": "Sturnella neglecta",
    "red_tailed_hawk": "Buteo jamaicensis",
    "california_quail": "Callipepla californica",
    "american_crow": "Corvus brachyrhynchos",
    "great_horned_owl": "Bubo virginianus"
}

# Filter for United States only (Quality filtering now happens in Python)
COUNTRY_FILTER = 'cnt:"United States"'

def ensure_dir(directory):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def download_file(url, filepath):
    """Downloads a file from a URL to a specific path."""
    try:
        # Xeno-canto sometimes returns //url, force https
        if url.startswith("//"):
            url = "https:" + url
            
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def get_recordings_metadata(scientific_name):
    """Fetches recording metadata using API v3."""
    # Query: Species AND Country (Quality tags removed to avoid 'AND' logic conflict)
    query_str = f'sp:"{scientific_name}" {COUNTRY_FILTER}'
    
    params = {
        'query': query_str,
        'key': API_KEY,
        # Request more than 100 because we will filter some out based on quality
        'per_page': 500 
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching metadata for {scientific_name}: {e}")
        return None

def main():
    print(f"Starting Xeno-canto Downloader (API v3 - Fixed Logic)")
    print(f"Target: {TARGET_SAMPLES} samples per species (Quality A or B)")
    print(f"Output: {OUTPUT_DIR.absolute()}\n")

    ensure_dir(OUTPUT_DIR)

    for folder_name, scientific_name in SPECIES.items():
        species_dir = OUTPUT_DIR / folder_name
        ensure_dir(species_dir)
        
        # Check existing files
        existing_files = list(species_dir.glob("*.mp3"))
        current_count = len(existing_files)
        
        if current_count >= TARGET_SAMPLES:
            print(f"✓ {folder_name}: Already have {current_count} files. Skipping.")
            continue

        print(f"Fetching metadata for: {folder_name} ({scientific_name})...")
        data = get_recordings_metadata(scientific_name)

        if not data or 'recordings' not in data:
            print(f"  No recordings found or API error for {folder_name}.")
            continue
            
        recordings = data['recordings']
        print(f"  Found {len(recordings)} total recordings. Filtering for Quality A/B...")

        downloaded_in_session = 0
        
        for rec in recordings:
            # Stop if we hit target
            if current_count >= TARGET_SAMPLES:
                break

            # === PYTHON-SIDE QUALITY FILTER ===
            # This implements "Quality A OR Quality B"
            if rec.get('q') not in ['A', 'B']:
                continue

            file_id = rec.get('id')
            file_url = rec.get('file')
            
            # Construct filename
            filename = f"{folder_name}_{file_id}.mp3"
            filepath = species_dir / filename

            if filepath.exists():
                continue

            print(f"  Downloading sample {current_count + 1}/{TARGET_SAMPLES} (ID: {file_id}, Q: {rec.get('q')})...", end='', flush=True)
            
            if download_file(file_url, filepath):
                print(" Done.")
                current_count += 1
                downloaded_in_session += 1
                time.sleep(0.5)
            else:
                print(" Failed.")

        if current_count < TARGET_SAMPLES:
            print(f"  Warning: Only found {current_count} valid samples (Quality A/B) for {folder_name}.")
        
        print(f"  Finished {folder_name}. Total files: {current_count}\n")
        
    print("All downloads complete.")

if __name__ == "__main__":
    main()