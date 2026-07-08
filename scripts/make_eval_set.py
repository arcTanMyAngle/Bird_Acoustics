#!/usr/bin/env python3
"""
make_eval_set.py - Stage an SD-card eval set for on-device parity (Phase 6).

Copies clips from data/processed (held-out species mix + background) into
eval_sd/eval/. Copy that eval/ folder to the microSD root, boot the board,
then run verify_device_parity.py on the eval_out.csv it produces.

Usage:
    uv run python scripts/make_eval_set.py [--n-per-class 3]
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="eval_sd/eval")
    parser.add_argument("--n-per-class", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    for cls_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        wavs = sorted(cls_dir.glob("*.wav"))
        step = max(1, len(wavs) // args.n_per_class)
        for wav in wavs[::step][:args.n_per_class]:
            shutil.copy(wav, out / wav.name)
            total += 1

    print(f"Staged {total} clips in {out} — copy the 'eval' folder to the SD card root.")


if __name__ == "__main__":
    main()
