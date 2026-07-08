#!/usr/bin/env python3
"""
verify_e2e_host.py - Host end-to-end check: C frontend -> int8 TFLite -> calibrated
decision, versus the pure-Python reference (torchaudio frontend -> same model).

Validates the entire device decision path except the esp-dsp FFT and I2S capture.
Gate: argmax agreement >= 99% between C-frontend and Python-frontend pipelines.

Usage (WSL):
    uv run python scripts/verify_e2e_host.py --model-dir models/exported_v5 \
        --calibration models/v5/calibration.json
"""

import argparse
import json
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


def run_tflite(interpreter, spec: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    s = inp["quantization_parameters"]["scales"][0]
    zp = int(inp["quantization_parameters"]["zero_points"][0])
    q = np.round(spec / s + zp).clip(-128, 127).astype(np.int8).reshape(1, 1, 40, 188)
    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"]).flatten().astype(np.float32)
    os_, ozp = out["quantization_parameters"]["scales"][0], \
        int(out["quantization_parameters"]["zero_points"][0])
    return (raw - ozp) * os_


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/exported_v5")
    parser.add_argument("--calibration", type=str, default="models/v5/calibration.json")
    parser.add_argument("--firmware-dir", type=str,
                        default="/mnt/c/Users/bornt/Desktop/Bird_Acoustics/firmware")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--n-clips", type=int, default=45)
    args = parser.parse_args()

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(Path(args.model_dir) / "bird_classifier_int8.tflite"))
    interpreter.allocate_tensors()
    with open(args.calibration) as f:
        cal = json.load(f)
    T, tau, bg = cal["temperature"], cal["tau"], cal["background_idx"]
    classes = cal["classes"]

    host_dir = Path(args.firmware_dir) / "host_test"
    subprocess.run(["make", "-C", str(host_dir)], check=True, capture_output=True)
    binary = host_dir / "parity_main"
    transform = AlignedMelSpectrogram()

    wavs = sorted(Path(args.data_dir).rglob("*.wav"))
    picked = wavs[::max(1, len(wavs) // args.n_clips)][:args.n_clips]

    agree = 0
    detections = {True: 0, False: 0}
    with tempfile.TemporaryDirectory() as td:
        for wav in picked:
            audio, sr = sf.read(wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.pad(audio, (0, max(0, CLIP - len(audio))))[:CLIP].astype(np.float32)

            fin, fout = Path(td) / "in.f32", Path(td) / "out.f32"
            audio.tofile(fin)
            subprocess.run([str(binary), str(fin), str(fout)], check=True)
            c_spec = np.fromfile(fout, dtype=np.float32)
            py_spec = transform(torch.from_numpy(audio).unsqueeze(0))[0].numpy().ravel()

            lc = run_tflite(interpreter, c_spec)
            lp = run_tflite(interpreter, py_spec)
            pc, pp = int(lc.argmax()), int(lp.argmax())
            if pc == pp:
                agree += 1
            else:
                print(f"  DISAGREE {wav.name}: C={classes[pc]} vs py={classes[pp]}")

            p = torch.softmax(torch.from_numpy(lc) / T, dim=0).numpy()
            is_bird_label = "background" not in wav.parts[-2]
            detected = pc != bg and p[pc] >= tau
            detections[is_bird_label] += int(detected)

    rate = agree / len(picked)
    print(f"\nC-vs-Python pipeline argmax agreement: {agree}/{len(picked)} ({rate:.1%})")
    print(f"Detections fired: {detections[True]} on bird clips, "
          f"{detections[False]} on background clips")
    if rate < 0.99:
        print("❌ GATE FAILED")
        return 1
    print("✅ GATE PASSED: full host pipeline consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
