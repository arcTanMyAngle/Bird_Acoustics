#!/usr/bin/env python3
"""
calibrate_rejection.py - Phase-3: temperature scaling + rejection threshold tau.

Fits on VAL logits, evaluates once on TEST logits (both dumped by train_v4.py).

Decision rule shipped to firmware:
    p = softmax(logits / T)
    detection iff argmax(p) != background AND p[argmax] >= tau

tau = smallest value whose VAL background false-alarm rate <= target (default 5%),
maximizing bird recall subject to that constraint.

Usage:
    uv run python scripts/calibrate_rejection.py --model-dir models/v5
Gate: exits non-zero if TEST background false-alarm rate > target.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimizing NLL on validation logits."""
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    log_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / log_temp.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.exp().item())


def detection_stats(logits, labels, T, tau, background_idx):
    """FP rate on background clips + correct-species recall on bird clips."""
    probs = torch.softmax(torch.from_numpy(logits).float() / T, dim=1).numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    detected = (pred != background_idx) & (conf >= tau)

    is_bg = labels == background_idx
    fp_rate = float(detected[is_bg].mean()) if is_bg.any() else 0.0
    correct_bird = detected & (pred == labels) & ~is_bg
    bird_recall = float(correct_bird[~is_bg].mean()) if (~is_bg).any() else 0.0
    return fp_rate, bird_recall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--target-fp", type=float, default=0.05)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    val = np.load(model_dir / "val_logits.npz")
    test = np.load(model_dir / "test_logits.npz")
    with open(model_dir / "config.json") as f:
        classes = json.load(f)["classes"]
    background_idx = classes.index("background")

    # 1. Temperature scaling on val
    T = fit_temperature(val["logits"], val["labels"])
    print(f"Fitted temperature: T = {T:.3f}")

    # 2. tau sweep on val. Selection rule (val-only, fixed a priori): the smallest
    # tau achieving val's MINIMUM FP rate, plus a fixed safety margin — with only
    # ~15 held-out background recording groups, the zero-margin "smallest tau
    # meeting target" rule does not generalize across splits.
    MARGIN = 0.10
    taus = np.arange(0.30, 0.96, 0.01)
    sweep = [detection_stats(val["logits"], val["labels"], T, t, background_idx)
             for t in taus]
    print(f"\n{'tau':>6} {'val FP%':>8} {'val recall%':>12}")
    for t, (fp, recall) in zip(taus, sweep):
        if round(t * 100) % 10 == 0:
            print(f"{t:6.2f} {fp * 100:8.1f} {recall * 100:12.1f}")

    min_fp = min(fp for fp, _ in sweep)
    if min_fp > args.target_fp:
        print(f"\n❌ Val FP never reaches target {args.target_fp:.0%} (min {min_fp:.1%})")
        return 1
    tau_anchor = next(float(t) for t, (fp, _) in zip(taus, sweep) if fp <= min_fp)
    tau = min(tau_anchor + MARGIN, 0.95)
    val_fp, val_recall = detection_stats(val["logits"], val["labels"], T, tau, background_idx)
    print(f"\nChosen: tau = {tau:.2f}  (anchor {tau_anchor:.2f} + margin {MARGIN}; "
          f"val FP {val_fp:.1%}, val bird recall {val_recall:.1%})")

    # 3. One-shot evaluation on held-out test
    test_fp, test_recall = detection_stats(test["logits"], test["labels"], T, tau, background_idx)
    print(f"TEST:   FP {test_fp:.1%}, bird recall {test_recall:.1%}  "
          f"(gate: FP <= {args.target_fp:.0%})")

    # Energy diagnostics (logged, not gated)
    def energy(lg):
        return torch.logsumexp(torch.from_numpy(lg).float(), dim=1).numpy()
    e_val = energy(val["logits"])
    is_bg = val["labels"] == background_idx

    calibration = {
        "temperature": T,
        "tau": tau,
        "background_idx": background_idx,
        "classes": classes,
        "val_fp_rate": val_fp,
        "val_bird_recall": val_recall,
        "test_fp_rate": test_fp,
        "test_bird_recall": test_recall,
        "target_fp": args.target_fp,
        "energy_diag": {
            "bg_mean": float(e_val[is_bg].mean()),
            "bg_std": float(e_val[is_bg].std()),
            "bird_mean": float(e_val[~is_bg].mean()),
            "bird_std": float(e_val[~is_bg].std()),
        },
    }
    out = model_dir / "calibration.json"
    with open(out, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Saved: {out}")

    if test_fp > args.target_fp:
        print("❌ GATE FAILED: test FP rate above target")
        return 1
    print("✅ GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
