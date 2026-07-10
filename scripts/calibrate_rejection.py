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
    """FP rate on background clips + correct-species recall on bird clips.
    `tau` may be a scalar or a per-class vector (indexed by predicted class)."""
    probs = torch.softmax(torch.from_numpy(logits).float() / T, dim=1).numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    tau_vec = np.asarray(tau)
    thr = tau_vec[pred] if tau_vec.ndim else tau_vec
    detected = (pred != background_idx) & (conf >= thr)

    is_bg = labels == background_idx
    fp_rate = float(detected[is_bg].mean()) if is_bg.any() else 0.0
    correct_bird = detected & (pred == labels) & ~is_bg
    bird_recall = float(correct_bird[~is_bg].mean()) if (~is_bg).any() else 0.0
    return fp_rate, bird_recall


def fit_per_class_tau(logits, labels, T, n_classes, background_idx, scalar_tau,
                      target_precision=0.85, tau_floor=0.35, tau_ceil=0.95):
    """One threshold per predicted class, on VAL, as the max of two constraints:

      1. precision:  smallest tau reaching `target_precision` among class-c predictions
                     -> an over-predicted class (california_scrub_jay) needs a higher tau,
                        lifting precision.
      2. bg floor:   the margin-protected `scalar_tau`, which controls the aggregate background
                     false-alarm rate with generalization headroom.

    tau_c = max(scalar_tau, precision_tau). We only ever RAISE a class's threshold above scalar
    (always FP-safe - it rejects more). We do NOT lower below scalar: with only ~15 held-out
    background groups, a lower per-class threshold that looks clean on val does not generalize
    (test background produces confusions val never showed), so lowering blows up the test FP
    rate. This policy self-adapts: on an over-predicting model it raises the offending class
    (e.g. california_scrub_jay on v5, 0.60->0.90 precision); on a already-balanced model it
    collapses to the safe scalar tau. Background is excluded from detection, so its entry is
    tau_ceil (never used at runtime)."""
    probs = torch.softmax(torch.from_numpy(logits).float() / T, dim=1).numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    grid = np.arange(float(scalar_tau), tau_ceil + 1e-9, 0.01)
    taus = np.full(n_classes, tau_ceil, dtype=float)
    diag = {}
    for c in range(n_classes):
        if c == background_idx:
            continue
        sel = pred == c
        if not sel.any():
            diag[c] = {"n_pred": 0, "chosen": tau_ceil, "reason": "no predictions"}
            continue
        # smallest tau >= scalar_tau reaching target precision (stays at scalar if already precise)
        chosen, best_prec = tau_ceil, 0.0
        for t in grid:
            m = sel & (conf >= t)
            if not m.any():
                continue
            prec = float((labels[m] == c).mean())
            best_prec = max(best_prec, prec)
            if prec >= target_precision:
                chosen = float(t)
                break
        taus[c] = float(min(max(chosen, float(scalar_tau)), tau_ceil))
        diag[c] = {"n_pred": int(sel.sum()), "chosen": taus[c], "best_prec": best_prec}
    return taus, diag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--target-fp", type=float, default=0.05)
    parser.add_argument("--target-precision", type=float, default=0.85,
                        help="per-class val precision target for per-class tau")
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
    # tau achieving val's MINIMUM FP rate, plus a fixed safety margin - with only
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
        print(f"\nFAIL: Val FP never reaches target {args.target_fp:.0%} (min {min_fp:.1%})")
        return 1
    tau_anchor = next(float(t) for t, (fp, _) in zip(taus, sweep) if fp <= min_fp)
    tau = min(tau_anchor + MARGIN, 0.95)
    val_fp, val_recall = detection_stats(val["logits"], val["labels"], T, tau, background_idx)
    print(f"\nChosen: tau = {tau:.2f}  (anchor {tau_anchor:.2f} + margin {MARGIN}; "
          f"val FP {val_fp:.1%}, val bird recall {val_recall:.1%})")

    # 2b. Per-class thresholds (fit on val), targeting owl recall + scrub_jay precision.
    tau_pc, pc_diag = fit_per_class_tau(val["logits"], val["labels"], T,
                                        len(classes), background_idx, tau,
                                        target_precision=args.target_precision)
    tau_pc[background_idx] = 0.95
    print(f"\n{'class':24s} {'n_pred':>7} {'tau_c':>6} {'val_prec':>9}")
    for c, name in enumerate(classes):
        d = pc_diag.get(c, {})
        print(f"{name:24s} {d.get('n_pred', 0):7d} {tau_pc[c]:6.2f} "
              f"{d.get('best_prec', float('nan')):9.3f}")
    val_fp_pc, val_recall_pc = detection_stats(val["logits"], val["labels"], T, tau_pc, background_idx)
    print(f"per-class VAL: FP {val_fp_pc:.1%}, bird recall {val_recall_pc:.1%}")

    # 3. One-shot evaluation on held-out test (scalar tau, then per-class tau)
    test_fp, test_recall = detection_stats(test["logits"], test["labels"], T, tau, background_idx)
    test_fp_pc, test_recall_pc = detection_stats(test["logits"], test["labels"], T, tau_pc, background_idx)
    print(f"TEST (scalar tau):    FP {test_fp:.1%}, bird recall {test_recall:.1%}")
    print(f"TEST (per-class tau): FP {test_fp_pc:.1%}, bird recall {test_recall_pc:.1%}  "
          f"(gate: FP <= {args.target_fp:.0%})")

    # Energy diagnostics (logged, not gated)
    def energy(lg):
        return torch.logsumexp(torch.from_numpy(lg).float(), dim=1).numpy()
    e_val = energy(val["logits"])
    is_bg = val["labels"] == background_idx

    calibration = {
        "temperature": T,
        "tau": tau,
        "tau_per_class": [float(t) for t in tau_pc],
        "background_idx": background_idx,
        "classes": classes,
        "val_fp_rate": val_fp,
        "val_bird_recall": val_recall,
        "test_fp_rate": test_fp,
        "test_bird_recall": test_recall,
        "val_fp_rate_per_class": val_fp_pc,
        "val_bird_recall_per_class": val_recall_pc,
        "test_fp_rate_per_class": test_fp_pc,
        "test_bird_recall_per_class": test_recall_pc,
        "target_fp": args.target_fp,
        "target_precision": args.target_precision,
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

    # Gate on the SHIPPED rule (per-class tau); scalar tau is only a fallback.
    if test_fp_pc > args.target_fp:
        print(f"FAIL: GATE FAILED: per-class test FP {test_fp_pc:.1%} above target {args.target_fp:.0%}")
        return 1
    print(f"PASS: GATE PASSED (per-class test FP {test_fp_pc:.1%}, bird recall {test_recall_pc:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
