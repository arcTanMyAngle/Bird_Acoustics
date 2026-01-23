#!/usr/bin/env python3
"""
download_xenocanto.py - Download bird audio from Xeno-canto API v3

Downloads high-quality (A/B rated) recordings for California bird species.
Filters to United States recordings only.

Usage:
    uv run python scripts/download_xenocanto.py
    uv run python scripts/download_xenocanto.py --target 150 --species california_scrub_jay,mourning_dove
"""

import os
import argparse
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
DEFAULT_TARGET_SAMPLES = 100

# Species Mapping (Folder Name -> Scientific Name)
# Original 5 species
SPECIES_CORE = {
    "western_meadowlark": "Sturnella neglecta",
    "red_tailed_hawk": "Buteo jamaicensis",
    "california_quail": "Callipepla californica",
    "american_crow": "Corvus brachyrhynchos",
    "great_horned_owl": "Bubo virginianus",
}

# Expanded species for Bay Area coverage
SPECIES_EXPANDED = {
    "mourning_dove": "Zenaida macroura",
    "yellow_billed_magpie": "Pica nuttalli",
    "red_winged_blackbird": "Agelaius phoeniceus",
    "marsh_wren": "Cistothorus palustris",
    "california_scrub_jay": "Aphelocoma californica",
    "northern_mockingbird": "Mimus polyglottos",
    "killdeer": "Charadrius vociferus",
}

# Combined dictionary
SPECIES_ALL = {**SPECIES_CORE, **SPECIES_EXPANDED}

# Recommended 3-class expansion (acoustically distinct, common in Bay Area)
SPECIES_RECOMMENDED_EXPANSION = {
    "mourning_dove": "Zenaida macroura",
    "california_scrub_jay": "Aphelocoma californica",
    "killdeer": "Charadrius vociferus",
}

# Filter for United States only
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
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def get_recordings_metadata(scientific_name, per_page=500):
    """Fetches recording metadata using API v3."""
    query_str = f'sp:"{scientific_name}" {COUNTRY_FILTER}'

    params = {
        "query": query_str,
        "key": API_KEY,
        "per_page": per_page,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching metadata for {scientific_name}: {e}")
        return None


def download_species(species_dict, output_dir, target_samples):
    """Download recordings for given species dictionary."""
    ensure_dir(output_dir)

    for folder_name, scientific_name in species_dict.items():
        species_dir = output_dir / folder_name
        ensure_dir(species_dir)

        # Check existing files
        existing_files = list(species_dir.glob("*.mp3"))
        current_count = len(existing_files)

        if current_count >= target_samples:
            print(f"✓ {folder_name}: Already have {current_count} files. Skipping.")
            continue

        print(f"Fetching metadata for: {folder_name} ({scientific_name})...")
        data = get_recordings_metadata(scientific_name)

        if not data or "recordings" not in data:
            print(f"  No recordings found or API error for {folder_name}.")
            continue

        recordings = data["recordings"]
        print(f"  Found {len(recordings)} total recordings. Filtering for Quality A/B...")

        downloaded_in_session = 0

        for rec in recordings:
            # Stop if we hit target
            if current_count >= target_samples:
                break

            # Python-side quality filter: A or B only
            if rec.get("q") not in ["A", "B"]:
                continue

            file_id = rec.get("id")
            file_url = rec.get("file")

            # Construct filename
            filename = f"{folder_name}_{file_id}.mp3"
            filepath = species_dir / filename

            if filepath.exists():
                continue

            print(
                f"  Downloading sample {current_count + 1}/{target_samples} "
                f"(ID: {file_id}, Q: {rec.get('q')})...",
                end="",
                flush=True,
            )

            if download_file(file_url, filepath):
                print(" Done.")
                current_count += 1
                downloaded_in_session += 1
                time.sleep(0.5)
            else:
                print(" Failed.")

        if current_count < target_samples:
            print(
                f"  Warning: Only found {current_count} valid samples (Quality A/B) for {folder_name}."
            )

        print(f"  Finished {folder_name}. Total files: {current_count}\n")


def main():
    parser = argparse.ArgumentParser(description="Download bird audio from Xeno-canto")
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_SAMPLES,
        help=f"Target samples per species (default: {DEFAULT_TARGET_SAMPLES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated list of species to download (default: all core species)",
    )
    parser.add_argument(
        "--include-expanded",
        action="store_true",
        help="Include all expanded species (12 total)",
    )
    parser.add_argument(
        "--recommended-only",
        action="store_true",
        help="Download only recommended 3-class expansion (dove, scrub jay, killdeer)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine which species to download
    if args.species:
        # User-specified subset
        requested = [s.strip() for s in args.species.split(",")]
        species_to_download = {k: v for k, v in SPECIES_ALL.items() if k in requested}
        missing = set(requested) - set(species_to_download.keys())
        if missing:
            print(f"Warning: Unknown species ignored: {missing}")
            print(f"Available: {list(SPECIES_ALL.keys())}")
    elif args.recommended_only:
        species_to_download = {**SPECIES_CORE, **SPECIES_RECOMMENDED_EXPANSION}
    elif args.include_expanded:
        species_to_download = SPECIES_ALL
    else:
        species_to_download = SPECIES_CORE

    print(f"Xeno-canto Downloader (API v3)")
    print(f"Target: {args.target} samples per species (Quality A or B)")
    print(f"Species: {len(species_to_download)}")
    print(f"Output: {output_dir.absolute()}\n")

    for name, sci in species_to_download.items():
        print(f"  - {name}: {sci}")
    print()

    download_species(species_to_download, output_dir, args.target)
    print("All downloads complete.")


if __name__ == "__main__":
    main()