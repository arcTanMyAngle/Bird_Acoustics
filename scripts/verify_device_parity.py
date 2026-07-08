#!/usr/bin/env python3
"""
verify_device_parity.py - Phase-6 gate: device logits vs host litert reference.

After the board processes /sdcard/eval/ (eval mode), copy eval_out.csv off the
SD card and run:

    uv run python scripts/verify_device_parity.py \
        --device-csv /path/to/eval_out.csv --wav-dir eval_sd/eval

Gate: argmax agreement >= 99%, mean |logit Δ| within ~2 int8 LSB (0.10).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset_v3 import AlignedMelSpectrogram

CLIP, HOP = 48000, 16000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-csv", type=str, required=True)
    parser.add_argument("--wav-dir", type=str, default="eval_sd/eval")
    parser.add_argument("--model", type=str,
                        default="models/exported_v5/bird_classifier_int8.tflite")
    args = parser.parse_args()

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_s = inp["quantization_parameters"]["scales"][0]
    in_zp = int(inp["quantization_parameters"]["zero_points"][0])
    out_s = out["quantization_parameters"]["scales"][0]
    out_zp = int(out["quantization_parameters"]["zero_points"][0])
    transform = AlignedMelSpectrogram()

    device = defaultdict(dict)
    with open(args.device_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_logits = sum(1 for h in header if h.startswith("logit_"))
        for row in reader:
            device[row[0]][int(row[1])] = np.array([float(x) for x in row[2:2 + n_logits]])

    agree = total = 0
    diffs = []
    for fname, windows in sorted(device.items()):
        audio, sr = sf.read(Path(args.wav_dir) / fname, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for w, dev_logits in sorted(windows.items()):
            seg = audio[w * HOP:w * HOP + CLIP]
            if len(seg) < CLIP:
                continue
            spec = transform(torch.from_numpy(seg).unsqueeze(0))[0].numpy()
            q = np.round(spec / in_s + in_zp).clip(-128, 127).astype(np.int8)
            interpreter.set_tensor(inp["index"], q.reshape(1, 1, 40, 188))
            interpreter.invoke()
            host = (interpreter.get_tensor(out["index"]).flatten().astype(np.float32)
                    - out_zp) * out_s
            total += 1
            agree += int(host.argmax() == dev_logits.argmax())
            diffs.append(np.abs(host - dev_logits).mean())

    rate = agree / max(total, 1)
    mae = float(np.mean(diffs)) if diffs else float("nan")
    print(f"windows compared: {total}")
    print(f"argmax agreement: {rate:.1%}   mean |logit Δ|: {mae:.4f} "
          f"({mae / out_s:.1f} int8 LSB)")
    if rate < 0.99 or mae > 0.10:
        print("❌ GATE FAILED")
        return 1
    print("✅ GATE PASSED: device matches host reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
