#!/usr/bin/env python3
"""
train_v4.py - Enhanced Training with Hard Negative Mining & Noise Rejection Validation

Key improvements over v3:
1. Hard negative mining: upweight difficult samples
2. Noise rejection test: dedicated evaluation for background rejection
3. Per-class threshold tuning based on validation
4. SNR-aware metrics: track detection at various noise levels
5. 9-class support for expanded California bird set

Usage:
    # Standard training
    uv run python scripts/train_v4.py --data-dir data/augmented --epochs 100
    
    # With hard negative mining
    uv run python scripts/train_v4.py --data-dir data/augmented --epochs 100 --hard-negative-mining
    
    # Full evaluation mode
    uv run python scripts/train_v4.py --data-dir data/augmented --epochs 100 --full-eval
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_v3 import (
    create_dataloaders_v3, 
    BirdAudioDatasetV3, 
    NoiseRejectionTestSet,
    HardNegativeMiner,
    BACKGROUND_CLASS,
)


# =============================================================================
# MODEL (9-class compatible)
# =============================================================================

class BirdClassifierCNN(nn.Module):
    """
    CNN for bird classification, compatible with 6-9 classes.
    ~65K parameters for 9 classes.
    """
    
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    hard_negative_miner: Optional[HardNegativeMiner] = None,
) -> Tuple[float, float]:
    """Train for one epoch with optional hard negative mining."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        # Handle both indexed and non-indexed batches
        if len(batch) == 3:
            inputs, labels, indices = batch
        else:
            inputs, labels = batch
            indices = None
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update hard negative miner
        if hard_negative_miner is not None and indices is not None:
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = probs.max(dim=1)
                hard_negative_miner.update(indices, predictions.cpu(), confidences.cpu(), labels.cpu())
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return running_loss / total, 100. * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return predictions with confidence scores."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        running_loss / total, 
        100. * correct / total, 
        np.array(all_preds), 
        np.array(all_labels),
        np.array(all_probs),
    )


def evaluate_noise_rejection(
    model: nn.Module,
    noise_test_set: NoiseRejectionTestSet,
    device: torch.device,
    background_idx: int,
) -> Dict:
    """Evaluate noise rejection performance."""
    print("\n" + "=" * 60)
    print("Noise Rejection Evaluation")
    print("=" * 60)
    
    results = noise_test_set.evaluate(model, device, background_idx)
    
    print(f"\nPure Background Rejection Rate: {results['background_rejection_rate']:.1f}%")
    print(f"False Positive Rate: {results['false_positive_rate']:.1f}%")
    
    print("\nBird Detection by SNR:")
    for snr in sorted(results['snr_performance'].keys()):
        perf = results['snr_performance'][snr]
        print(f"  SNR {snr:2d} dB: {perf['accuracy']:.1f}% ({perf['correct']}/{perf['total']})")
    
    return results


def compute_optimal_thresholds(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    classes: List[str],
    target_precision: float = 0.9,
) -> Dict[str, float]:
    """
    Compute per-class confidence thresholds targeting specific precision.
    
    This helps tune thresholds for deployment where we want to minimize
    false positives for certain classes.
    """
    thresholds = {}
    
    for i, cls in enumerate(classes):
        # Binary classification: this class vs all others
        y_true = (all_labels == i).astype(int)
        y_scores = all_probs[:, i]
        
        if y_true.sum() == 0:
            thresholds[cls] = 0.5
            continue
        
        # Compute precision-recall curve
        precision, recall, threshold_values = precision_recall_curve(y_true, y_scores)
        
        # Find threshold achieving target precision
        valid_idx = np.where(precision >= target_precision)[0]
        if len(valid_idx) > 0:
            # Take the threshold with highest recall among those with sufficient precision
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            if best_idx < len(threshold_values):
                thresholds[cls] = float(threshold_values[best_idx])
            else:
                thresholds[cls] = 0.5
        else:
            # Can't achieve target precision, use median
            thresholds[cls] = 0.5
    
    return thresholds


def dump_logits(model: nn.Module, loader: DataLoader, device: torch.device, path: Path):
    """Save raw logits+labels for post-hoc calibration (temperature / tau fitting)."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    np.savez(path, logits=np.concatenate(logits), labels=np.concatenate(labels))
    print(f"Saved: {path}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_history(history: Dict, save_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'orange', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'orange', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Background rejection (if available)
    if 'bg_rejection_rate' in history and history['bg_rejection_rate']:
        axes[1, 1].plot(epochs[:len(history['bg_rejection_rate'])], 
                        history['bg_rejection_rate'], 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Background Rejection Rate (%)')
        axes[1, 1].set_title('Noise Rejection Performance')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
    else:
        axes[1, 1].text(0.5, 0.5, 'No noise rejection data', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Noise Rejection Performance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, classes: List[str], save_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_snr_performance(noise_results: Dict, save_path: Path):
    """Plot bird detection accuracy vs SNR."""
    snr_levels = sorted(noise_results['snr_performance'].keys())
    accuracies = [noise_results['snr_performance'][snr]['accuracy'] for snr in snr_levels]
    
    plt.figure(figsize=(8, 6))
    plt.plot(snr_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=90, color='g', linestyle='--', label='90% target')
    plt.axhline(y=noise_results['background_rejection_rate'], 
                color='r', linestyle='--', label=f'BG rejection: {noise_results["background_rejection_rate"]:.1f}%')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bird Detection Accuracy (%)')
    plt.title('Detection Performance vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    noise_test_set: NoiseRejectionTestSet,
    num_epochs: int,
    device: torch.device,
    output_dir: Path,
    class_weights: torch.Tensor,
    classes: List[str],
    lr: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 20,
    min_delta: float = 0.5,
    hard_negative_miner: Optional[HardNegativeMiner] = None,
    eval_noise_every: int = 10,
    test_loader: Optional[DataLoader] = None,
) -> Dict:
    """Full training loop with noise rejection evaluation."""
    
    background_idx = classes.index(BACKGROUND_CLASS) if BACKGROUND_CLASS in classes else -1
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Classes: {len(classes)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Noise test samples: {len(noise_test_set)}")
    print(f"Background class index: {background_idx}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Initial LR: {lr}")
    print(f"Hard negative mining: {'enabled' if hard_negative_miner else 'disabled'}")
    print(f"Device: {device}")
    
    # Loss function. Class-weighted CE is the ONLY imbalance correction (the weighted
    # sampler is disabled in main() — using both double-corrects and skews the boundary).
    # Label smoothing tempers overconfidence so the softmax threshold has headroom.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr / 100
    )
    
    # History
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
        'bg_rejection_rate': [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_noise_results = None
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, num_epochs, hard_negative_miner
        )
        
        # Validate
        val_loss, val_acc, preds, labels, probs = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"Loss: {val_loss:.3f}", end="")
        
        # Evaluate noise rejection periodically
        if (epoch + 1) % eval_noise_every == 0 or epoch == num_epochs - 1:
            noise_results = noise_test_set.evaluate(model, device, background_idx)
            history['bg_rejection_rate'].append(noise_results['background_rejection_rate'])
            print(f" | BG Rej: {noise_results['background_rejection_rate']:.1f}%", end="")
        
        # Check for improvement
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': classes,
                'num_classes': len(classes),
                'config': {
                    'n_mels': 40,
                    'n_fft': 512,
                    'hop_length': 256,
                    'sample_rate': 16000,
                    'duration': 3.0,
                }
            }, output_dir / "best_model.pth")
            print(" ✓ BEST")
        else:
            patience_counter += 1
            print()
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Log hard negative mining stats
        if hard_negative_miner is not None and (epoch + 1) % 10 == 0:
            stats = hard_negative_miner.get_statistics()
            print(f"  [HNM] Misclassified: {stats['n_misclassified']}, "
                  f"Low conf: {stats['n_low_confidence']}, "
                  f"Mean difficulty: {stats['mean_difficulty']:.2f}")
    
    # Training complete
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    _, final_acc, final_preds, final_labels, final_probs = validate(
        model, val_loader, criterion, device
    )
    
    # Final noise rejection evaluation
    print("\n" + "=" * 60)
    print("Final Noise Rejection Evaluation")
    print("=" * 60)
    final_noise_results = evaluate_noise_rejection(model, noise_test_set, device, background_idx)
    
    # Held-out TEST evaluation — touched exactly once, after model selection
    test_metrics = {}
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("Held-out TEST Evaluation (honest metrics)")
        print("=" * 60)
        _, test_acc, test_preds, test_labels, _ = validate(model, test_loader, criterion, device)
        print(f"TEST accuracy: {test_acc:.2f}%")
        print(classification_report(
            test_labels, test_preds, target_names=classes,
            labels=list(range(len(classes))), zero_division=0
        ))
        plot_confusion_matrix(test_preds, test_labels, classes,
                              output_dir / "confusion_matrix_test.png")
        test_metrics = {
            "test_acc": test_acc,
            "report": classification_report(
                test_labels, test_preds, target_names=classes,
                labels=list(range(len(classes))), zero_division=0, output_dict=True
            ),
        }
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Raw logits for post-hoc calibration (Phase 3: temperature + tau)
    dump_logits(model, val_loader, device, output_dir / "val_logits.npz")
    if test_loader is not None:
        dump_logits(model, test_loader, device, output_dir / "test_logits.npz")

    # Compute optimal thresholds
    print("\nComputing optimal per-class thresholds (target 90% precision)...")
    thresholds = compute_optimal_thresholds(final_probs, final_labels, classes, target_precision=0.9)
    print("Optimal thresholds:")
    for cls, thresh in thresholds.items():
        print(f"  {cls}: {thresh:.3f}")
    
    # Save plots
    plot_training_history(history, output_dir / "training_curves.png")
    print(f"\nSaved: {output_dir / 'training_curves.png'}")
    
    plot_confusion_matrix(final_preds, final_labels, classes, output_dir / "confusion_matrix.png")
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    
    plot_snr_performance(final_noise_results, output_dir / "snr_performance.png")
    print(f"Saved: {output_dir / 'snr_performance.png'}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        final_labels, 
        final_preds, 
        target_names=classes,
        labels=list(range(len(classes))),
        zero_division=0
    ))
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save thresholds
    with open(output_dir / "thresholds.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # Save noise results
    # Remove non-serializable items
    noise_results_save = {k: v for k, v in final_noise_results.items() if k != 'predictions'}
    with open(output_dir / "noise_rejection_results.json", 'w') as f:
        json.dump(noise_results_save, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'num_classes': len(classes),
        'best_val_acc': best_val_acc,
        'thresholds': thresholds,
        'config': checkpoint['config']
    }, output_dir / "final_model.pth")
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_acc': final_acc,
        'test_metrics': test_metrics,
        'noise_rejection': final_noise_results,
        'thresholds': thresholds,
        'history': history,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train bird classifier v4 with noise rejection")
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    parser.add_argument("--output-dir", type=str, default="models/v4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--hard-negative-mining", action="store_true", help="Enable hard negative mining")
    parser.add_argument("--eval-noise-every", type=int, default=10, help="Evaluate noise rejection every N epochs")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, noise_test_set, classes, class_weights, miner = create_dataloaders_v3(
        str(data_dir),
        batch_size=args.batch_size,
        seed=args.seed,
        augment_train=not args.no_augment,
        use_weighted_sampler=False,  # single imbalance correction: weighted CE in train()
        enable_hard_negative_mining=args.hard_negative_mining,
    )
    
    num_classes = len(classes)
    print(f"\nClasses ({num_classes}): {classes}")
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    
    # Sanity check: verify data shapes and labels
    print("\nData sanity check...")
    
    # Check multiple batches to see class distribution
    from collections import Counter
    train_label_counts = Counter()
    val_label_counts = Counter()
    
    for i, batch in enumerate(train_loader):
        labels = batch[1] if len(batch) >= 2 else batch[1]
        for l in labels.tolist():
            train_label_counts[l] += 1
        if i >= 10:  # Check first 10 batches
            break
    
    for i, batch in enumerate(val_loader):
        labels = batch[1] if len(batch) >= 2 else batch[1]
        for l in labels.tolist():
            val_label_counts[l] += 1
        if i >= 10:
            break
    
    print(f"  Train class distribution (first 10 batches):")
    for cls_idx in range(num_classes):
        count = train_label_counts.get(cls_idx, 0)
        cls_name = classes[cls_idx][:12]
        print(f"    {cls_idx} ({cls_name}): {count}")
    
    print(f"  Val class distribution (first 10 batches):")
    missing_in_val = []
    for cls_idx in range(num_classes):
        count = val_label_counts.get(cls_idx, 0)
        cls_name = classes[cls_idx][:12]
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"    {cls_idx} ({cls_name}): {count} {status}")
        if count == 0:
            missing_in_val.append(classes[cls_idx])
    
    if missing_in_val:
        print(f"\n  ⚠ WARNING: Classes missing from validation: {missing_in_val}")
        print(f"  This will cause training to fail!")
        raise ValueError(f"Validation set missing classes: {missing_in_val}")
    
    # Quick shape check
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0]
    print(f"\n  Spectrogram shape: {sample_x.shape}")
    print(f"  Spectrogram stats: mean={sample_x.mean():.3f}, std={sample_x.std():.3f}")
    
    # Create model
    model = BirdClassifierCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        noise_test_set=noise_test_set,
        num_epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
        classes=classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        hard_negative_miner=miner,
        eval_noise_every=args.eval_noise_every,
        test_loader=test_loader,
    )
    
    # Save config
    config = {
        'args': vars(args),
        'results': {
            'best_val_acc': results['best_val_acc'],
            'best_epoch': results['best_epoch'],
            'test_acc': results['test_metrics'].get('test_acc'),
            'background_rejection_rate': results['noise_rejection']['background_rejection_rate'],
            'false_positive_rate': results['noise_rejection']['false_positive_rate'],
        },
        'classes': classes,
        'thresholds': results['thresholds'],
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nKey metrics:")
    print(f"  Validation accuracy: {results['best_val_acc']:.1f}%")
    print(f"  Background rejection: {results['noise_rejection']['background_rejection_rate']:.1f}%")
    print(f"  False positive rate: {results['noise_rejection']['false_positive_rate']:.1f}%")
    
    print("\nNext steps:")
    print(f"  1. Review confusion matrix: {output_dir}/confusion_matrix.png")
    print(f"  2. Review SNR performance: {output_dir}/snr_performance.png")
    print(f"  3. Export model: uv run python scripts/export_v3.py --model-path {output_dir}/best_model.pth")


if __name__ == "__main__":
    main()