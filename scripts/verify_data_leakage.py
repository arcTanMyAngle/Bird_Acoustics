#!/usr/bin/env python3
"""
verify_data_leakage.py - Verify if train/val split has data leakage

This script checks if clips from the same original recording are present
in both train and validation sets, which would inflate validation accuracy.

Usage:
    python verify_data_leakage.py --data-dir data/augmented
"""

import argparse
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import random_split, Subset
import random

def analyze_leakage(data_dir: Path, val_split: float = 0.2, seed: int = 42):
    """
    Analyze potential data leakage in the current split strategy.
    """
    print("=" * 60)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 60)
    
    # Collect all samples and group by source recording
    samples = []
    groups = defaultdict(list)
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        for audio_file in class_dir.glob("*.wav"):
            samples.append((audio_file, class_dir.name))
            
            # Extract group key (original recording)
            # Expected format: {class}_{file_idx}_{clip_idx}.wav
            name = audio_file.stem
            parts = name.rsplit('_', 1)  # Split off the last underscore
            
            if len(parts) == 2 and parts[1].isdigit():
                group_key = parts[0]  # e.g., "crow_0001"
            else:
                group_key = name  # Fallback: use full name
            
            groups[group_key].append(len(samples) - 1)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total recording groups: {len(groups)}")
    print(f"  Average clips per group: {len(samples) / len(groups):.2f}")
    
    # Analyze group sizes
    group_sizes = [len(indices) for indices in groups.values()]
    multi_clip_groups = [k for k, v in groups.items() if len(v) > 1]
    
    print(f"\n  Single-clip groups: {len(groups) - len(multi_clip_groups)}")
    print(f"  Multi-clip groups: {len(multi_clip_groups)}")
    print(f"  Max clips per group: {max(group_sizes)}")
    
    # Simulate the current random_split
    print("\n" + "-" * 60)
    print("SIMULATING CURRENT SPLIT (random_split at clip level)")
    print("-" * 60)
    
    total_size = len(samples)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Recreate the exact split from dataset.py
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_indices = set(indices[val_size:])
    val_indices = set(indices[:val_size])
    
    # Check for leakage
    leaked_groups = []
    for group_key, group_indices in groups.items():
        if len(group_indices) > 1:
            in_train = [i for i in group_indices if i in train_indices]
            in_val = [i for i in group_indices if i in val_indices]
            
            if in_train and in_val:
                leaked_groups.append({
                    'group': group_key,
                    'total': len(group_indices),
                    'in_train': len(in_train),
                    'in_val': len(in_val)
                })
    
    if leaked_groups:
        print(f"\n⚠️  LEAKAGE DETECTED!")
        print(f"  Groups with clips in BOTH train and val: {len(leaked_groups)}")
        
        # Show examples
        print(f"\n  Example leaked groups:")
        for lg in leaked_groups[:10]:
            print(f"    {lg['group']}: {lg['in_train']} in train, {lg['in_val']} in val")
        
        if len(leaked_groups) > 10:
            print(f"    ... and {len(leaked_groups) - 10} more")
        
        # Calculate impact
        total_leaked_samples = sum(lg['total'] for lg in leaked_groups)
        val_leaked = sum(lg['in_val'] for lg in leaked_groups)
        
        print(f"\n  Impact:")
        print(f"    Leaked samples in validation: {val_leaked} ({100*val_leaked/val_size:.1f}% of val set)")
        print(f"    These samples have 'sister' clips in training set")
        print(f"    Model may memorize recording-specific features instead of species features")
    else:
        print("\n✅ No leakage detected (no groups have clips in both splits)")
    
    # Propose grouped split
    print("\n" + "-" * 60)
    print("PROPOSED FIX: Split by recording group")
    print("-" * 60)
    
    group_keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)
    
    n_val_groups = int(len(group_keys) * val_split)
    val_groups = set(group_keys[:n_val_groups])
    train_groups = set(group_keys[n_val_groups:])
    
    new_train_indices = [i for g in train_groups for i in groups[g]]
    new_val_indices = [i for g in val_groups for i in groups[g]]
    
    print(f"\n  Grouped split statistics:")
    print(f"    Training groups: {len(train_groups)} ({len(new_train_indices)} samples)")
    print(f"    Validation groups: {len(val_groups)} ({len(new_val_indices)} samples)")
    print(f"    No clips from same recording can appear in both sets")
    
    # Verify no leakage in grouped split
    print("\n  Verification: checking for leakage in grouped split...")
    for group_key, group_indices in groups.items():
        in_new_train = [i for i in group_indices if i in new_train_indices]
        in_new_val = [i for i in group_indices if i in new_val_indices]
        if in_new_train and in_new_val:
            print(f"    ❌ ERROR: Group {group_key} still has leakage!")
            break
    else:
        print("    ✅ Grouped split has no leakage")
    
    return leaked_groups


def generate_grouped_split_code():
    """Print the code needed to fix the dataset.py file."""
    print("\n" + "=" * 60)
    print("RECOMMENDED CODE CHANGE FOR dataset.py")
    print("=" * 60)
    
    code = '''
def create_dataloaders_grouped(data_dir, batch_size=32, val_split=0.2, num_workers=4, seed=42):
    """Create train and validation dataloaders with GROUPED split to prevent leakage."""
    from collections import defaultdict
    import random
    
    dataset = BirdAudioDataset(data_dir)
    
    # Group samples by source recording
    groups = defaultdict(list)
    for idx, (path, label) in enumerate(dataset.samples):
        name = path.stem
        parts = name.rsplit('_', 1)
        group_key = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
        groups[group_key].append(idx)
    
    # Split at group level
    group_keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)
    
    n_val_groups = int(len(group_keys) * val_split)
    
    val_indices = [idx for key in group_keys[:n_val_groups] for idx in groups[key]]
    train_indices = [idx for key in group_keys[n_val_groups:] for idx in groups[key]]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Grouped split: {len(train_indices)} train, {len(val_indices)} val")
    print(f"  From {len(group_keys) - n_val_groups} train groups, {n_val_groups} val groups")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, dataset.classes, dataset.get_class_weights()
'''
    print(code)


def main():
    parser = argparse.ArgumentParser(description="Verify data leakage in bird audio dataset")
    parser.add_argument("--data-dir", type=str, default="data/augmented",
                        help="Path to augmented data directory")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (should match training)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please provide the correct path with --data-dir")
        return 1
    
    leaked_groups = analyze_leakage(data_dir, args.val_split, args.seed)
    generate_grouped_split_code()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if leaked_groups:
        print(f"\n❌ DATA LEAKAGE CONFIRMED")
        print(f"   {len(leaked_groups)} recording groups have clips in both train and val")
        print(f"   This explains high validation accuracy ({96.69}%) vs poor real-world performance")
        print(f"\n   ACTION REQUIRED:")
        print(f"   1. Update dataset.py with grouped split function above")
        print(f"   2. Retrain model")
        print(f"   3. Expect validation accuracy to drop to 75-85%")
        print(f"   4. But real-world accuracy should now MATCH validation")
    else:
        print(f"\n✅ No obvious data leakage detected")
        print(f"   Issue may be preprocessing mismatch or calibration")
    
    return 0


if __name__ == "__main__":
    exit(main())
