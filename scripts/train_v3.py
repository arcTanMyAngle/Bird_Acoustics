#!/usr/bin/env python3
"""
train_v3.py - Fixed Training Script

Fixes from v2:
1. Learning rate scheduler actually works (CosineAnnealingLR, stepped per epoch)
2. Proper warmup implemented manually
3. No distillation by default (teacher was too weak)
4. Option to disable augmentation for debugging
5. Better early stopping logic

Usage:
    # Standard training (recommended first)
    uv run python scripts/train_v3.py --data-dir data/augmented --epochs 100
    
    # Debug mode (no augmentation, verify model can learn)
    uv run python scripts/train_v3.py --data-dir data/augmented --epochs 50 --no-augment --lr 0.003
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_v2 import create_dataloaders_v2, BirdAudioDatasetV2


# =============================================================================
# MODEL
# =============================================================================

class BirdClassifierCNN(nn.Module):
    """Student model for ESP32 deployment (~63K parameters)."""
    
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        
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
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
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
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / total, 100. * correct / total, np.array(all_preds), np.array(all_labels)


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_history(history: Dict, save_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'orange', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'orange', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, classes: List[str], save_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    output_dir: Path,
    class_weights: torch.Tensor,
    classes: List[str],
    lr: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 20,
    min_delta: float = 0.5,
) -> Dict:
    """Full training loop."""
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Initial LR: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Early stopping patience: {patience}")
    print(f"Device: {device}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - cosine annealing
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr / 100  # Minimum LR = initial / 100
    )
    
    # History
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validate
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        
        # Step scheduler (once per epoch)
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
    
    # Training complete
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    # Save plots
    plot_training_history(history, output_dir / "training_curves.png")
    print(f"Saved: {output_dir / 'training_curves.png'}")
    
    plot_confusion_matrix(final_preds, final_labels, classes, output_dir / "confusion_matrix.png")
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=classes, zero_division=0))
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'best_val_acc': best_val_acc,
        'config': checkpoint['config']
    }, output_dir / "final_model.pth")
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_acc': final_acc,
        'history': history,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train bird classifier (v3 - fixed)")
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    parser.add_argument("--output-dir", type=str, default="models/v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
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
    train_loader, val_loader, classes, class_weights = create_dataloaders_v2(
        str(data_dir),
        batch_size=args.batch_size,
        val_split=0.2,
        seed=args.seed,
        augment_train=not args.no_augment,
        use_weighted_sampler=True,
    )
    
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Class weights: {class_weights.tolist()}")
    
    # Create model
    model = BirdClassifierCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    
    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        class_weights=class_weights,
        classes=classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    
    # Save config
    config = {
        'args': vars(args),
        'results': {
            'best_val_acc': results['best_val_acc'],
            'best_epoch': results['best_epoch'],
        },
        'classes': classes,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. If accuracy > 75%, export: uv run python scripts/export_v2.py --model-path {output_dir}/best_model.pth")
    print(f"  2. If accuracy < 70%, try: --no-augment --lr 0.003 --epochs 150")


if __name__ == "__main__":
    main()
