# ~/bird-detection/scripts/lottery_ticket_pruning.py
"""
Lottery Ticket Hypothesis Implementation for Bird Acoustic Detection
Implements Iterative Magnitude Pruning (IMP) to find winning tickets.

Reference: Frankle & Carlin (2019) "The Lottery Ticket Hypothesis: 
Finding Sparse, Trainable Neural Networks"

Usage:
    uv run python scripts/lottery_ticket_pruning.py --target-sparsity 0.8
    uv run python scripts/lottery_ticket_pruning.py --target-sparsity 0.9 --rounds 15
"""

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

from dataset import create_dataloaders
from model import get_model
from train import train_epoch, validate


class LotteryTicketFinder:
    """
    Implements Iterative Magnitude Pruning to find winning tickets.
    
    The algorithm:
    1. Initialize network with weights θ₀
    2. Train to completion, obtaining θⱼ
    3. Prune p% lowest magnitude weights
    4. Reset remaining weights to θ₀
    5. Repeat until target sparsity reached
    """
    
    def __init__(
        self,
        model_name: str = "standard",
        data_dir: str = "data/augmented",
        output_dir: str = "models/lottery_ticket",
        target_sparsity: float = 0.8,
        prune_rate: float = 0.2,
        epochs_per_round: int = 30,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = None,
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_sparsity = target_sparsity
        self.prune_rate = prune_rate
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.lr = lr
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data
        print(f"Loading data from {self.data_dir}...")
        self.train_loader, self.val_loader, self.classes, self.class_weights = \
            create_dataloaders(str(self.data_dir), batch_size=self.batch_size)
        self.num_classes = len(self.classes)
        
        # Initialize model and save original weights
        print(f"Initializing {self.model_name} model...")
        self.model = get_model(self.model_name, num_classes=self.num_classes)
        self.model.to(self.device)
        
        # Store original initialization (θ₀)
        self.initial_weights = copy.deepcopy(self.model.state_dict())
        
        # Tracking
        self.history = {
            "rounds": [],
            "sparsity": [],
            "val_accuracy": [],
            "train_accuracy": [],
            "parameters": [],
        }
    
    def get_prunable_layers(self):
        """Returns list of (module, name) tuples for layers to prune."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers.append((module, "weight"))
        return layers
    
    def calculate_sparsity(self):
        """Calculate current global sparsity of the model."""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    zero_params += (mask == 0).sum().item()
                else:
                    weight = module.weight
                    total_params += weight.numel()
                    zero_params += (weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def count_nonzero_parameters(self):
        """Count non-zero parameters in the model."""
        total = 0
        nonzero = 0
        
        for name, param in self.model.named_parameters():
            total += param.numel()
            if 'weight_mask' in name:
                continue
            nonzero += (param != 0).sum().item()
        
        return nonzero, total
    
    def apply_global_unstructured_pruning(self, amount):
        """Apply global unstructured magnitude pruning."""
        layers = self.get_prunable_layers()
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            layers,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    
    def reset_to_initial_weights(self):
        """Reset remaining (unpruned) weights to their initial values θ₀."""
        current_masks = {}
        
        # Extract current masks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    current_masks[name] = module.weight_mask.clone()
        
        # Remove pruning reparametrization temporarily
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        # Load initial weights
        self.model.load_state_dict(self.initial_weights, strict=False)
        
        # Reapply masks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in current_masks:
                    prune.custom_from_mask(module, 'weight', current_masks[name])
    
    def train_round(self, round_num):
        """Train the model for one pruning round."""
        # Setup optimizer and scheduler (fresh each round)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.epochs_per_round
        )
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device)
        )
        
        best_val_acc = 0
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Training for {self.epochs_per_round} epochs")
        print(f"Current sparsity: {self.calculate_sparsity()*100:.1f}%")
        print(f"{'='*60}")
        
        for epoch in range(self.epochs_per_round):
            train_loss, train_acc = train_epoch(
                self.model, self.train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc, _, _ = validate(
                self.model, self.val_loader, criterion, self.device
            )
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        return best_val_acc, train_acc
    
    def find_winning_ticket(self, max_rounds: int = 20):
        """
        Main IMP loop to find winning ticket.
        
        Returns:
            dict: Results including final sparsity, accuracy, and winning ticket state
        """
        print("\n" + "="*60)
        print("LOTTERY TICKET HYPOTHESIS - ITERATIVE MAGNITUDE PRUNING")
        print("="*60)
        print(f"Target sparsity: {self.target_sparsity*100:.0f}%")
        print(f"Prune rate per round: {self.prune_rate*100:.0f}%")
        print(f"Epochs per round: {self.epochs_per_round}")
        print(f"Device: {self.device}")
        print("="*60)
        
        round_num = 0
        current_sparsity = 0
        
        while current_sparsity < self.target_sparsity and round_num < max_rounds:
            round_num += 1
            
            # Train the model
            val_acc, train_acc = self.train_round(round_num)
            
            # Calculate remaining amount to prune to avoid overshooting
            remaining_sparsity = self.target_sparsity - current_sparsity
            prune_amount = min(self.prune_rate, remaining_sparsity / (1 - current_sparsity))
            
            # Prune
            print(f"\nPruning {prune_amount*100:.1f}% of remaining weights...")
            self.apply_global_unstructured_pruning(prune_amount)
            
            # Reset to initial weights (keeping the mask)
            print("Resetting to initial weights θ₀...")
            self.reset_to_initial_weights()
            
            # Update sparsity
            current_sparsity = self.calculate_sparsity()
            nonzero, total = self.count_nonzero_parameters()
            
            # Log progress
            self.history["rounds"].append(round_num)
            self.history["sparsity"].append(current_sparsity)
            self.history["val_accuracy"].append(val_acc)
            self.history["train_accuracy"].append(train_acc)
            self.history["parameters"].append(nonzero)
            
            print(f"\n✓ Round {round_num} complete:")
            print(f"  Sparsity: {current_sparsity*100:.1f}%")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            print(f"  Non-zero params: {nonzero:,} / {total:,}")
        
        # Final training of winning ticket
        print("\n" + "="*60)
        print("FINAL TRAINING OF WINNING TICKET")
        print("="*60)
        final_val_acc, final_train_acc = self.train_round("Final")
        
        # Save results
        results = {
            "target_sparsity": self.target_sparsity,
            "achieved_sparsity": current_sparsity,
            "final_val_accuracy": final_val_acc,
            "final_train_accuracy": final_train_acc,
            "total_rounds": round_num,
            "classes": self.classes,
            "model_name": self.model_name,
            "history": self.history,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.save_winning_ticket(results)
        
        return results
    
    def save_winning_ticket(self, results):
        """Save the winning ticket model and results."""
        
        # Make pruning permanent for export
        model_export = copy.deepcopy(self.model)
        for name, module in model_export.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        # Save winning ticket state dict
        sparsity_pct = int(results["achieved_sparsity"] * 100)
        model_path = self.output_dir / f"winning_ticket_{sparsity_pct}pct.pth"
        
        torch.save({
            "model_state_dict": model_export.state_dict(),
            "classes": results["classes"],
            "sparsity": results["achieved_sparsity"],
            "val_accuracy": results["final_val_accuracy"],
            "config": {
                "model": self.model_name,
                "n_mels": 40,
                "sample_rate": 16000,
                "duration": 3.0,
            }
        }, model_path)
        
        print(f"\n✓ Saved winning ticket to {model_path}")
        
        # Save results JSON
        results_path = self.output_dir / f"lottery_ticket_results_{sparsity_pct}pct.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved results to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("WINNING TICKET FOUND!")
        print("="*60)
        print(f"Sparsity: {results['achieved_sparsity']*100:.1f}%")
        print(f"Validation Accuracy: {results['final_val_accuracy']:.2f}%")
        print(f"Pruning Rounds: {results['total_rounds']}")
        
        nonzero, total = self.count_nonzero_parameters()
        print(f"Parameters: {nonzero:,} / {total:,} ({nonzero/total*100:.1f}% remaining)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Find winning tickets using Iterative Magnitude Pruning"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/augmented",
        help="Path to augmented dataset"
    )
    parser.add_argument(
        "--model", type=str, default="standard", choices=["standard", "small"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.8,
        help="Target sparsity level (0.0 to 0.95)"
    )
    parser.add_argument(
        "--prune-rate", type=float, default=0.2,
        help="Fraction of weights to prune per round"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs per pruning round"
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Maximum number of pruning rounds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/lottery_ticket",
        help="Output directory for winning tickets"
    )
    
    args = parser.parse_args()
    
    # Validate target sparsity
    if not 0 < args.target_sparsity < 1:
        raise ValueError("Target sparsity must be between 0 and 1")
    
    # Initialize and run
    finder = LotteryTicketFinder(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_sparsity=args.target_sparsity,
        prune_rate=args.prune_rate,
        epochs_per_round=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    results = finder.find_winning_ticket(max_rounds=args.rounds)
    
    print("\n✓ Lottery Ticket identification complete!")
    print(f"  Use the winning ticket for deployment: {args.output_dir}/")


if __name__ == "__main__":
    main()
