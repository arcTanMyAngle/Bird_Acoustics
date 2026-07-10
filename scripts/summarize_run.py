#!/usr/bin/env python3
"""
summarize_run.py - compact, token-cheap digest of a models/<run>/ directory.

Context sees the digest (scalars + top-3 confusions); disk keeps the full 9x9 matrix.
Reads the artifacts train_v4.py / calibrate_rejection.py already emit:
  test_metrics.json  (classification report)   [required]
  test_logits.npz    (logits, labels)          [optional -> confusion cells]
  calibration.json   (temperature, tau, FP)    [optional]
  noise_rejection_results.json (SNR sweep)     [optional]
  config.json        (args, timestamp)         [optional]

Usage:
  uv run python scripts/summarize_run.py --model-dir models/v6
  uv run python scripts/summarize_run.py --model-dir models/v6 --append \\
      --run v6 --frontend 512/64/7000 --arch dw-sep --loss cb-focal --data cleaned+0dB

The two "target cells" are the classes this project tracks: great_horned_owl recall
(false-negative symptom) and california_scrub_jay precision (over-prediction symptom).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

OWL = "great_horned_owl"
JAY = "california_scrub_jay"
LEDGER = Path("docs/EXPERIMENTS.md")
LEDGER_COLS = ["run", "date", "frontend", "arch", "loss", "data",
               "val_mF1", "test_acc", "owl_R", "jay_P", "bg_FP@tau", "5dB_det", "ms", "notes"]


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def snr_map(noise):
    """Normalize either schema -> {snr_str: accuracy_pct}. Supports the flat
    'snr_accuracy' {snr: pct} and the nested 'snr_performance' {snr: {accuracy: pct}}."""
    if noise.get("snr_accuracy"):
        return {k: float(v) for k, v in noise["snr_accuracy"].items()}
    if noise.get("snr_performance"):
        return {k: float(v.get("accuracy", 0)) for k, v in noise["snr_performance"].items()}
    return {}


def top_confusions(model_dir, classes, k=3):
    """Top-k off-diagonal (true -> pred) cells from raw argmax(test logits)."""
    npz = model_dir / "test_logits.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    pred = d["logits"].argmax(1)
    true = d["labels"]
    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    cells = [(cm[t, p], classes[t], classes[p])
             for t in range(n) for p in range(n) if t != p and cm[t, p] > 0]
    cells.sort(reverse=True)
    return cells[:k]


def digest(model_dir: Path):
    metrics = _load(model_dir / "test_metrics.json")
    if metrics is None:
        raise SystemExit(f"no test_metrics.json in {model_dir}")
    rep = metrics["report"]
    cfg = _load(model_dir / "config.json") or {}
    cal = _load(model_dir / "calibration.json") or {}
    noise = _load(model_dir / "noise_rejection_results.json") or {}
    classes = cfg.get("classes") or [k for k in rep
                                      if k not in ("accuracy", "macro avg", "weighted avg")]

    macro = rep["macro avg"]
    lines = [f"# {model_dir.name}  ({cfg.get('timestamp', '?')[:10]})",
             f"test_acc {metrics['test_acc']:.2f}%   "
             f"macro P/R/F1 {macro['precision']:.3f}/{macro['recall']:.3f}/{macro['f1-score']:.3f}"]
    if OWL in rep:
        o = rep[OWL]
        lines.append(f"  {OWL:22s} P {o['precision']:.3f}  R {o['recall']:.3f}  (FN symptom)")
    if JAY in rep:
        j = rep[JAY]
        lines.append(f"  {JAY:22s} P {j['precision']:.3f}  R {j['recall']:.3f}  (FP-sink symptom)")
    if cal:
        lines.append(f"reject: T {cal.get('temperature', 0):.3f}  tau {cal.get('tau', 0)}  "
                     f"test_FP {cal.get('test_fp_rate', 0):.1%}  "
                     f"bird_recall {cal.get('test_bird_recall', 0):.1%}")
    snr = snr_map(noise)
    if snr:
        lines.append("SNR det%: " + "  ".join(f"{k}dB {v:.0f}" for k, v in sorted(
            snr.items(), key=lambda kv: float(kv[0]))))
    conf = top_confusions(model_dir, classes)
    if conf:
        lines.append("top confusions (true->pred):")
        lines += [f"  {n:4d}  {t} -> {p}" for n, t, p in conf]
    return lines, metrics, macro, rep, cal, noise, cfg


def ledger_row(args, metrics, macro, rep, cal, noise, cfg):
    def cell(x):
        return "" if x is None else x
    owl_r = f"{rep[OWL]['recall']:.3f}" if OWL in rep else ""
    jay_p = f"{rep[JAY]['precision']:.3f}" if JAY in rep else ""
    bg_fp = f"{cal.get('test_fp_rate', ''):.1%}" if cal.get("test_fp_rate") is not None else ""
    det5 = ""
    snr = snr_map(noise)
    if snr.get("5") is not None:
        det5 = f"{snr['5']:.0f}"
    vals = [args.run or cfg.get("output_dir", "?").split("/")[-1],
            datetime.now().strftime("%Y-%m-%d"),
            cell(args.frontend), cell(args.arch), cell(args.loss), cell(args.data),
            f"{macro['f1-score']:.3f}", f"{metrics['test_acc']:.2f}",
            owl_r, jay_p, bg_fp, det5, cell(args.ms), cell(args.notes)]
    return "| " + " | ".join(str(v) for v in vals) + " |"


def ensure_ledger():
    if LEDGER.exists():
        return
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(LEDGER_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in LEDGER_COLS) + " |"
    LEDGER.write_text(
        "# Experiment ledger\n\n"
        "One scalar row per run. Full confusion matrices live on disk in `models/<run>/` — "
        "never paste the 9x9 grid into a session; run `scripts/summarize_run.py` for the digest.\n"
        "`owl_R` = great_horned_owl recall (FN symptom); `jay_P` = california_scrub_jay precision "
        "(FP-sink symptom); `bg_FP@tau` = calibrated background false-alarm on test.\n\n"
        + header + "\n" + sep + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--append", action="store_true", help="append a row to docs/EXPERIMENTS.md")
    for f in ("run", "frontend", "arch", "loss", "data", "ms", "notes"):
        ap.add_argument(f"--{f}", default=None)
    args = ap.parse_args()

    lines, metrics, macro, rep, cal, noise, cfg = digest(Path(args.model_dir))
    print("\n".join(lines))

    if args.append:
        ensure_ledger()
        row = ledger_row(args, metrics, macro, rep, cal, noise, cfg)
        with LEDGER.open("a") as f:
            f.write(row + "\n")
        print(f"\nappended -> {LEDGER}")


if __name__ == "__main__":
    main()
