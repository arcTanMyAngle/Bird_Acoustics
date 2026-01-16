# ~/bird-detection/scripts/verify_winning_ticket.py
"""
Verify the actual sparsity and size of a winning ticket model.
This script properly counts zeros in the weight tensors.

Usage:
    uv run python scripts/verify_winning_ticket.py models/lottery_ticket/winning_ticket_79pct.pth
"""

import argparse
import torch
from pathlib import Path


def analyze_model(model_path: str):
    """Analyze sparsity and size of a saved model."""
    
    print("\n" + "=" * 60)
    print("WINNING TICKET VERIFICATION")
    print("=" * 60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    print(f"\nModel: {model_path}")
    print(f"Classes: {checkpoint.get('classes', 'N/A')}")
    print(f"Reported sparsity: {checkpoint.get('sparsity', 'N/A'):.2%}")
    print(f"Reported accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    # Analyze each layer
    print("\n" + "-" * 60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("-" * 60)
    print(f"{'Layer':<35} {'Total':>10} {'Zeros':>10} {'Sparsity':>10}")
    print("-" * 60)
    
    total_params = 0
    total_zeros = 0
    
    layer_data = []
    
    for name, param in state_dict.items():
        if 'weight' in name and 'bn' not in name.lower():  # Skip batch norm
            params = param.numel()
            zeros = (param == 0).sum().item()
            sparsity = zeros / params if params > 0 else 0
            
            total_params += params
            total_zeros += zeros
            
            layer_data.append({
                'name': name,
                'params': params,
                'zeros': zeros,
                'sparsity': sparsity
            })
            
            print(f"{name:<35} {params:>10,} {zeros:>10,} {sparsity:>10.1%}")
    
    # Global stats
    global_sparsity = total_zeros / total_params if total_params > 0 else 0
    nonzero_params = total_params - total_zeros
    
    print("-" * 60)
    print(f"{'TOTAL':<35} {total_params:>10,} {total_zeros:>10,} {global_sparsity:>10.1%}")
    
    # Size analysis
    print("\n" + "-" * 60)
    print("MODEL SIZE ANALYSIS")
    print("-" * 60)
    
    # Calculate sizes
    fp32_size = total_params * 4  # 4 bytes per float32
    fp32_nonzero_size = nonzero_params * 4
    int8_size = total_params * 1  # 1 byte per int8
    int8_nonzero_size = nonzero_params * 1
    
    # Sparse formats add ~4 bytes overhead per nonzero for index storage
    # But for deployment, zeros are typically just stored as dense with compression
    
    print(f"Original FP32 (dense):           {fp32_size / 1024:>8.1f} KB")
    print(f"Pruned FP32 (only nonzero):      {fp32_nonzero_size / 1024:>8.1f} KB  ({fp32_nonzero_size/fp32_size:.0%})")
    print(f"Original INT8 (dense):           {int8_size / 1024:>8.1f} KB")
    print(f"Pruned INT8 (only nonzero):      {int8_nonzero_size / 1024:>8.1f} KB  ({int8_nonzero_size/fp32_size:.0%})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Actual sparsity:     {global_sparsity:.2%}")
    print(f"  Non-zero parameters: {nonzero_params:,} / {total_params:,}")
    print(f"  Compression ratio:   {total_params / nonzero_params:.1f}x (from pruning alone)")
    print(f"  With INT8 quant:     {fp32_size / int8_nonzero_size:.1f}x (pruning + quantization)")
    print("=" * 60)
    
    return {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': global_sparsity,
        'layers': layer_data
    }


def compare_models(original_path: str, pruned_path: str):
    """Compare original and pruned models side by side."""
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    orig = analyze_model(original_path)
    pruned = analyze_model(pruned_path)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Original non-zero:  {orig['nonzero_params']:,}")
    print(f"  Pruned non-zero:    {pruned['nonzero_params']:,}")
    print(f"  Parameters removed: {orig['nonzero_params'] - pruned['nonzero_params']:,}")
    print(f"  Reduction:          {1 - pruned['nonzero_params']/orig['nonzero_params']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify winning ticket model")
    parser.add_argument("model_path", type=str, help="Path to winning ticket .pth file")
    parser.add_argument("--compare", type=str, help="Path to original model for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.model_path)
    else:
        analyze_model(args.model_path)
