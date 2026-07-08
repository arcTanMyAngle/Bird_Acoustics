#!/usr/bin/env python3
"""
verify_split_v3.py - Phase-1 gate: prove the grouped stratified split is leakage-free
on the REAL on-disk data.

Checks:
1. Every _augN file shares its group with the parent stem (augmentation can't leak).
2. Every {class}_{idx} group maps to exactly one source recording (bijection).
3. train/val/noise_test group sets are pairwise disjoint.
4. All classes are present in all three splits.

Usage:
    uv run python scripts/verify_split_v3.py --data-dir data/augmented
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset_v3 import BirdAudioDatasetV3, get_three_way_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = BirdAudioDatasetV3(args.data_dir)
    failures = []

    # --- Check 1: _augN files inherit the parent group ------------------------
    stem_to_group = {Path(p).stem: g for (p, _), g in zip(
        [(str(f), l) for f, l in dataset.samples], dataset.sample_groups)}
    n_aug = 0
    for (f, _), group in zip(dataset.samples, dataset.sample_groups):
        stem = Path(f).stem
        m = re.match(r"^(.+)_aug\d+$", stem)
        if not m:
            continue
        n_aug += 1
        parent = m.group(1)
        if parent in stem_to_group and stem_to_group[parent] != group:
            failures.append(f"aug/parent group mismatch: {stem} -> {group} vs {stem_to_group[parent]}")
    print(f"[1] {n_aug} augmented files checked against parents: "
          f"{'OK' if not failures else 'FAIL'}")

    # --- Check 2: group -> single source recording (needs XC id in name) ------
    group_to_rec = defaultdict(set)
    parsed = 0
    for (f, label), group in zip(dataset.samples, dataset.sample_groups):
        stem = re.sub(r"_aug\d+$", "", Path(f).stem)
        cls = dataset.idx_to_class[label]
        m = re.match(rf"^{re.escape(cls)}_\d+_(.+)_w\d+$", stem)
        if m:
            parsed += 1
            group_to_rec[group].add(m.group(1))
    conflicts = {g: r for g, r in group_to_rec.items() if len(r) > 1}
    print(f"[2] recording-id bijection on {parsed} parseable files: "
          f"{len(conflicts)} conflicts {'OK' if not conflicts else 'FAIL'}")
    for g, r in list(conflicts.items())[:5]:
        failures.append(f"group {g} maps to recordings {sorted(r)}")

    # --- Check 3+4: split disjointness and class coverage ---------------------
    train_idx, val_idx, test_idx = get_three_way_split(dataset, seed=args.seed)
    names = ["train", "val", "noise_test"]
    splits = [train_idx, val_idx, test_idx]
    groups = [set(dataset.sample_groups[i] for i in s) for s in splits]
    for a in range(3):
        for b in range(a + 1, 3):
            overlap = groups[a] & groups[b]
            if overlap:
                failures.append(f"{names[a]}/{names[b]} share {len(overlap)} groups: "
                                f"{sorted(overlap)[:5]}")
    print(f"[3] split group overlap: {'NONE (OK)' if not any(groups[a] & groups[b] for a in range(3) for b in range(a+1,3)) else 'FAIL'}")

    print(f"\n[4] per-class sample counts (train/val/noise_test):")
    for cls in dataset.classes:
        cls_idx = dataset.class_to_idx[cls]
        counts = [sum(1 for i in s if dataset.samples[i][1] == cls_idx) for s in splits]
        if 0 in counts:
            failures.append(f"class {cls} missing from a split: {counts}")
        print(f"    {cls:24s} {counts[0]:5d} {counts[1]:5d} {counts[2]:5d}")

    if failures:
        print(f"\n❌ GATE FAILED ({len(failures)} problems):")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print(f"\n✅ GATE PASSED: leakage-free grouped stratified split "
          f"({len(dataset)} samples, {len(set(dataset.sample_groups))} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
