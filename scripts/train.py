# ~/bird-detection/scripts/train.py

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from dataset import create_dataloaders
from model import get_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_labels


def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(preds, labels, classes, save_path):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train bird classifier")
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    parser.add_argument("--model", type=str, default="standard", choices=["standard", "small"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, classes, class_weights = create_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size
    )
    
    num_classes = len(classes)
    print(f"\nClasses ({num_classes}): {classes}")
    
    # Create model
    model = get_model(args.model, num_classes=num_classes).to(device)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    best_val_acc = 0.0
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes
            }, output_dir / "best_model.pth")
            print(f"  ✓ Saved best model (acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "config": {
            "model": args.model,
            "n_mels": 40,
            "sample_rate": 16000,
            "duration": 3.0
        }
    }, output_dir / "final_model.pth")
    
    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plot_training_history(history, output_dir / "training_curves.png")
    plot_confusion_matrix(preds, labels, classes, output_dir / "confusion_matrix.png")
    
    # Save class mapping
    with open(output_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    
    print(f"\nOutputs saved to {output_dir}/")
    print("  - best_model.pth")
    print("  - final_model.pth")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - classes.json")
    print("  - history.json")


if __name__ == "__main__":
    main()
