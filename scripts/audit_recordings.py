#!/usr/bin/env python3
"""
audit_recordings.py - Phase-1 data engineering: per-recording noise/quality audit.

Computes acoustic-quality metrics per 3 s window, aggregates to the source-recording
(group) level using the SAME group key as dataset_v3.py / verify_split_v3.py
({class}_{first-token-after-class-prefix}), and assigns a triage label:

  keep          clean enough to train on as-is
  denoise       recoverable but noisy/windy -> spectral-gate + per-class HPF (clean_recordings.py)
  drop          noise-dominated, unrecoverable as signal
  harvest_noise recording is mixed; some windows are pure noise -> route those to background/hard-neg

Metrics per window (frontend-aligned: 16 kHz, n_fft 512, hop 256):
  snr_db     10*log10(P90 / P10) of per-frame energy   (dynamic range; low => uniform noise)
  flatness   mean spectral flatness (Wiener entropy)    (high => broadband/noise-like)
  wind_frac  fraction of energy below 100 Hz            (rumble/wind; owl fundamental is ~300 Hz)
  centroid   spectral centroid in Hz                    (off-target / species-overlap sniff)

Thresholds are CLI-tunable; the script prints the metric distribution so they can be calibrated
to this dataset rather than guessed. Writes:
  data/audit/recording_quality.json   (per-group metrics + triage + per-window noise flags)
  docs/DATA_AUDIT.md                   (per-class digest + worst-10 recordings)

Usage:
  uv run python scripts/audit_recordings.py --data-dir data/processed
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
N_FFT = 512
HOP = 256
LOW_HZ = 100.0
BACKGROUND = "background"


def group_key(stem: str, cls: str) -> str:
    """Match dataset_v3.py: strip class prefix, take first token, re-prefix with class."""
    if stem.startswith(cls + "_"):
        rec = stem[len(cls) + 1:].split("_")[0]
    else:
        rec = stem
    return f"{cls}_{rec}"


def window_metrics(y: np.ndarray) -> dict:
    import librosa
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size))
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2  # power (F, T)
    frame_e = S.sum(axis=0) + 1e-12
    p10, p90 = np.percentile(frame_e, [10, 90])
    snr_db = 10.0 * np.log10(p90 / (p10 + 1e-12))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=np.sqrt(S))))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    low = S[freqs < LOW_HZ, :].sum()
    wind_frac = float(low / (S.sum() + 1e-12))
    centroid = float(np.sum(freqs[:, None] * S) / (S.sum() + 1e-12))
    return {"snr_db": float(snr_db), "flatness": flat,
            "wind_frac": wind_frac, "centroid": centroid}


def triage(m: dict, cls: str, args) -> str:
    if cls == BACKGROUND:
        return "keep"                      # background is meant to be noise
    noisy = m["flatness"] > args.flat_hi
    flat_signal = m["snr_db"] < args.snr_lo
    windy = m["wind_frac"] > args.wind_hi
    if noisy and flat_signal:
        return "drop"
    if noisy or windy:
        return "denoise"
    return "keep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--flat-hi", type=float, default=0.35, help="flatness above => noise-like")
    ap.add_argument("--snr-lo", type=float, default=4.0, help="snr_db below => uniform noise")
    ap.add_argument("--wind-hi", type=float, default=0.25, help="sub-100Hz fraction => wind")
    ap.add_argument("--out", default="data/audit/recording_quality.json")
    args = ap.parse_args()

    import librosa
    data_dir = Path(args.data_dir)
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    groups = defaultdict(lambda: {"cls": None, "windows": []})  # group_key -> metrics list
    all_snr, all_flat, all_wind = [], [], []

    for cls in classes:
        files = sorted((data_dir / cls).glob("*.wav"))
        print(f"{cls:24s} {len(files):5d} windows ...", flush=True)
        for f in files:
            y, _ = librosa.load(f, sr=SR, mono=True)
            m = window_metrics(y)
            m["file"] = f.name
            m["is_noise_window"] = bool(m["flatness"] > args.flat_hi and m["snr_db"] < args.snr_lo)
            g = group_key(f.stem, cls)
            groups[g]["cls"] = cls
            groups[g]["windows"].append(m)
            all_snr.append(m["snr_db"]); all_flat.append(m["flatness"]); all_wind.append(m["wind_frac"])

    # Aggregate to group level (median over windows) + triage
    report = {}
    triage_counts = defaultdict(lambda: defaultdict(int))
    for g, d in groups.items():
        w = d["windows"]
        agg = {k: float(np.median([x[k] for x in w])) for k in ("snr_db", "flatness", "wind_frac", "centroid")}
        label = triage(agg, d["cls"], args)
        n_noise = sum(x["is_noise_window"] for x in w)
        if label == "keep" and n_noise > 0 and d["cls"] != BACKGROUND:
            label = "harvest_noise"        # clean overall, but has pure-noise windows to extract
        report[g] = {"class": d["cls"], "n_windows": len(w), "n_noise_windows": n_noise,
                     "triage": label, **agg,
                     "noise_windows": [x["file"] for x in w if x["is_noise_window"]]}
        triage_counts[d["cls"]][label] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    # --- console distribution (for threshold calibration) ---
    def pct(a):
        return " ".join(f"p{p}={np.percentile(a, p):.2f}" for p in (10, 50, 90))
    print("\n=== metric distribution (calibrate thresholds to these) ===")
    print(f"  snr_db   {pct(all_snr)}   (--snr-lo {args.snr_lo})")
    print(f"  flatness {pct(all_flat)}   (--flat-hi {args.flat_hi})")
    print(f"  wind     {pct(all_wind)}   (--wind-hi {args.wind_hi})")

    # --- markdown digest ---
    lines = ["# Data audit — recording quality & noise triage", "",
             f"Source: `{args.data_dir}` · {len(report)} recordings · "
             f"thresholds flat>{args.flat_hi}, snr<{args.snr_lo} dB, wind>{args.wind_hi}", "",
             "## Per-class triage", "",
             "| class | keep | denoise | harvest | drop | med snr_db | med flat | med wind |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cls in classes:
        gs = [r for r in report.values() if r["class"] == cls]
        tc = triage_counts[cls]
        med = lambda k: np.median([r[k] for r in gs]) if gs else 0
        lines.append(f"| {cls} | {tc['keep']} | {tc['denoise']} | {tc['harvest_noise']} | "
                     f"{tc['drop']} | {med('snr_db'):.1f} | {med('flatness'):.2f} | {med('wind_frac'):.2f} |")
    worst = sorted(report.items(), key=lambda kv: kv[1]["snr_db"])[:10]
    lines += ["", "## Worst 10 recordings (lowest dynamic range)", "",
              "| group | class | triage | snr_db | flat | wind | noise win |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for g, r in worst:
        lines.append(f"| {g} | {r['class']} | {r['triage']} | {r['snr_db']:.1f} | "
                     f"{r['flatness']:.2f} | {r['wind_frac']:.2f} | {r['n_noise_windows']}/{r['n_windows']} |")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/DATA_AUDIT.md").write_text("\n".join(lines) + "\n")

    total = defaultdict(int)
    for cls in triage_counts:
        for k, v in triage_counts[cls].items():
            total[k] += v
    print(f"\ntriage totals: {dict(total)}")
    print(f"wrote {out} and docs/DATA_AUDIT.md")


if __name__ == "__main__":
    main()
