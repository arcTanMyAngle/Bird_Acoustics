#!/usr/bin/env python3
"""
clean_recordings.py - Phase-1 data engineering: apply the audit triage.

Reads data/audit/recording_quality.json (from audit_recordings.py) and produces a cleaned
copy of data/processed that the rest of the pipeline (augment_audio.py -> dataset_v3.py) then
consumes unchanged. Actions per recording:

  keep           straight copy (no processing)
  denoise        spectral-gating denoise (noisereduce) + per-class high-pass to kill wind rumble
                 (owl/mourning_dove use a LOW 80 Hz cutoff to protect their ~300 Hz fundamentals;
                  other classes 150 Hz)
  drop           excluded from the cleaned set (its noise windows are still harvested)
  harvest_noise  clean windows copied through; the recording's pure-noise windows are routed to
                 the background class as extra reject data

Harvested noise is written into the cleaned `background/` class with a unique per-source group key
(`background_h{NNNN}`), so the leakage-invariant bijection still holds (one source -> one group).

Grouping is preserved: every cleaned file keeps its original filename, so dataset_v3.py's
{class}_{idx} group key is unchanged for keep/denoise recordings.

Usage:
  uv run python scripts/clean_recordings.py --data-dir data/processed \\
      --audit data/audit/recording_quality.json --out data/cleaned
Then: rm -rf data/augmented && uv run python scripts/augment_audio.py --mix-noise ... \\
      (point INPUT_DIR at data/cleaned) and re-run verify_split_v3.py.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 16000
HPF_CUTOFF = {"great_horned_owl": 80.0, "mourning_dove": 80.0}   # low cut to protect low fundamentals
HPF_DEFAULT = 150.0
BACKGROUND = "background"


def highpass(y: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff / (SR / 2.0), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def denoise(y: np.ndarray, cls: str, prop_decrease: float) -> np.ndarray:
    import noisereduce as nr
    out = nr.reduce_noise(y=y, sr=SR, stationary=False, prop_decrease=prop_decrease)
    out = highpass(out, HPF_CUTOFF.get(cls, HPF_DEFAULT))
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak * 0.99
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--audit", default="data/audit/recording_quality.json")
    ap.add_argument("--out", default="data/cleaned")
    ap.add_argument("--prop-decrease", type=float, default=0.8, help="noisereduce strength 0..1")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    report = json.loads(Path(args.audit).read_text())

    stats = {"keep": 0, "denoise": 0, "drop": 0, "harvest_noise": 0,
             "files_copied": 0, "files_denoised": 0, "files_harvested": 0, "files_dropped": 0}
    harvest_idx = 0

    for group, r in sorted(report.items()):
        cls = r["class"]
        triage = r["triage"]
        stats[triage] += 1
        cls_out = out_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        noise_set = set(r.get("noise_windows", []))
        files = sorted((data_dir / cls).glob(f"{group}_*.wav")) or \
            [data_dir / cls / f for f in _files_for_group(data_dir / cls, group)]

        if triage in ("keep", "denoise"):
            for f in files:
                if triage == "keep":
                    shutil.copy2(f, cls_out / f.name)
                    stats["files_copied"] += 1
                else:
                    y, _ = sf.read(f, dtype="float32")
                    sf.write(cls_out / f.name, denoise(y, cls, args.prop_decrease), SR)
                    stats["files_denoised"] += 1

        elif triage == "harvest_noise":
            bg_out = out_dir / BACKGROUND
            bg_out.mkdir(parents=True, exist_ok=True)
            harvested_any = False
            for f in files:
                if f.name in noise_set:
                    harvested_any = True
            hkey = f"h{harvest_idx:04d}" if harvested_any else None
            wn = 0
            for f in files:
                if f.name in noise_set:
                    y, _ = sf.read(f, dtype="float32")
                    name = f"{BACKGROUND}_{hkey}_{cls}_w{wn}.wav"
                    sf.write(bg_out / name, y, SR)
                    wn += 1
                    stats["files_harvested"] += 1
                else:
                    shutil.copy2(f, cls_out / f.name)
                    stats["files_copied"] += 1
            if harvested_any:
                harvest_idx += 1

        elif triage == "drop":
            bg_out = out_dir / BACKGROUND
            bg_out.mkdir(parents=True, exist_ok=True)
            hkey = f"h{harvest_idx:04d}"
            wn = 0
            for f in files:
                if f.name in noise_set:
                    y, _ = sf.read(f, dtype="float32")
                    sf.write(bg_out / f"{BACKGROUND}_{hkey}_{cls}_w{wn}.wav", y, SR)
                    wn += 1
                    stats["files_harvested"] += 1
                stats["files_dropped"] += 1
            if wn:
                harvest_idx += 1

    print("triage groups:", {k: stats[k] for k in ("keep", "denoise", "harvest_noise", "drop")})
    print("files:", {k: stats[k] for k in
                     ("files_copied", "files_denoised", "files_harvested", "files_dropped")})
    print(f"cleaned dataset -> {out_dir}")


def _files_for_group(cls_dir: Path, group: str):
    """Fallback matcher: files whose {class}_{idx} prefix equals the group key."""
    cls = cls_dir.name
    idx = group[len(cls) + 1:]
    return [f.name for f in cls_dir.glob("*.wav")
            if f.stem.startswith(cls + "_") and f.stem[len(cls) + 1:].split("_")[0] == idx]


if __name__ == "__main__":
    main()
