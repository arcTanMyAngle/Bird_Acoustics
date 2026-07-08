#!/usr/bin/env python3
"""
verify_frontend_parity.py - Phase-5 gate: the C frontend must reproduce
torchaudio's AlignedMelSpectrogram on real recordings before anything is flashed.

Builds firmware/host_test/parity_main (device frontend.c + reference FFT) with
gcc, runs it on N real clips, and compares against the training transform.

Gate: max |Δ| < 1e-2 on the z-scored spectrogram (≈0.2 int8 LSB).

Usage (WSL):
    uv run python scripts/verify_frontend_parity.py \
        --firmware-dir /mnt/c/Users/bornt/Desktop/Bird_Acoustics/firmware
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset_v3 import AlignedMelSpectrogram

CLIP = 48000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware-dir", type=str,
                        default="/mnt/c/Users/bornt/Desktop/Bird_Acoustics/firmware")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--n-clips", type=int, default=27)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()

    host_dir = Path(args.firmware_dir) / "host_test"
    subprocess.run(["make", "-C", str(host_dir)], check=True)
    binary = host_dir / "parity_main"

    transform = AlignedMelSpectrogram()
    data_dir = Path(args.data_dir)
    wavs = sorted(data_dir.rglob("*.wav"))
    step = max(1, len(wavs) // args.n_clips)
    picked = wavs[::step][:args.n_clips]

    worst = 0.0
    worst_file = None
    with tempfile.TemporaryDirectory() as td:
        for wav in picked:
            audio, sr = sf.read(wav, dtype="float32")
            assert sr == 16000, wav
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) < CLIP:
                audio = np.pad(audio, (0, CLIP - len(audio)))
            audio = audio[:CLIP].astype(np.float32)

            fin, fout = Path(td) / "in.f32", Path(td) / "out.f32"
            audio.tofile(fin)
            subprocess.run([str(binary), str(fin), str(fout)], check=True)
            c_spec = np.fromfile(fout, dtype=np.float32).reshape(40, 188)

            ref = transform(torch.from_numpy(audio).unsqueeze(0))[0].numpy()
            diff = float(np.abs(c_spec - ref).max())
            status = "ok" if diff < args.tolerance else "FAIL"
            print(f"  {wav.name:55s} max|Δ| = {diff:.2e}  {status}")
            if diff > worst:
                worst, worst_file = diff, wav.name

    print(f"\nWorst case: {worst:.2e} ({worst_file}), tolerance {args.tolerance:.0e}")
    if worst >= args.tolerance:
        print("❌ GATE FAILED: C frontend diverges from torchaudio")
        return 1
    print("✅ GATE PASSED: C frontend matches training frontend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
