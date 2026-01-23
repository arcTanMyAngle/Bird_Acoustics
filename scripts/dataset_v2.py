#!/usr/bin/env python3
"""
dataset_v2.py - Fixed Dataset with Grouped Split and Aligned Preprocessing

Key fixes over original dataset.py:
1. Grouped train/val split to prevent data leakage
2. Preprocessing aligned exactly with firmware (including top_db clamp)
3. Optional augmentations for real-world robustness
4. Proper class balancing

Usage:
    from dataset_v2 import create_dataloaders_v2, BirdAudioDatasetV2
"""

import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf


# =============================================================================
# SPECAUGMENT AND AUGMENTATIONS
# =============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    https://arxiv.org/abs/1904.08779
    
    Applies time and frequency masking to spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor of shape (C, F, T) where F=n_mels, T=time_frames
        """
        if random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0:t0 + t] = 0
        
        return spec


class AudioAugment:
    """
    Audio-level augmentations applied before spectrogram extraction.
    Helps with real-world robustness.
    """
    
    def __init__(
        self,
        noise_snr_range: Tuple[float, float] = (10, 30),  # dB
        gain_range: Tuple[float, float] = (-6, 6),  # dB
        pitch_shift_range: Tuple[int, int] = (-2, 2),  # semitones
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        p_noise: float = 0.3,
        p_gain: float = 0.5,
        p_pitch: float = 0.0,  # Disabled by default (slow)
        p_stretch: float = 0.0,  # Disabled by default (slow)
    ):
        self.noise_snr_range = noise_snr_range
        self.gain_range = gain_range
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.p_noise = p_noise
        self.p_gain = p_gain
        self.p_pitch = p_pitch
        self.p_stretch = p_stretch
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR."""
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    def apply_gain(self, waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain in dB."""
        return waveform * (10 ** (gain_db / 20))
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Add noise
        if random.random() < self.p_noise:
            snr = random.uniform(*self.noise_snr_range)
            waveform = self.add_noise(waveform, snr)
        
        # Apply gain
        if random.random() < self.p_gain:
            gain = random.uniform(*self.gain_range)
            waveform = self.apply_gain(waveform, gain)
        
        # Clip to prevent overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


# =============================================================================
# ALIGNED MEL SPECTROGRAM TRANSFORM
# =============================================================================

class AlignedMelSpectrogram(nn.Module):
    """
    Mel spectrogram transform aligned with ESP32 firmware implementation.
    
    Key alignment points:
    1. Same n_fft, hop_length, n_mels
    2. Same top_db clamping (80 dB)
    3. Same per-sample normalization
    4. center=True (default) - firmware should match this
    
    Note: The firmware uses left-aligned framing. For best alignment,
    you could set center=False, but this changes the time axis length.
    We keep center=True and note that first/last few frames may differ slightly.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
        top_db: float = 80.0,
        center: bool = True,  # torchaudio default
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        
        # Mel spectrogram transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,  # Power spectrogram
            center=center,
            norm='slaney',
            mel_scale='htk',
        )
        
        # Amplitude to dB with top_db clamping
        # This matches: AmplitudeToDB(stype="power", top_db=80)
        self.amplitude_to_db = T.AmplitudeToDB(
            stype="power",
            top_db=top_db
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (C, T) audio tensor
        
        Returns:
            mel_spec_db_norm: (C, n_mels, n_frames) normalized mel spectrogram
        """
        # Compute mel spectrogram
        mel_spec = self.mel_spec(waveform)  # (C, n_mels, n_frames)
        
        # Convert to dB with top_db clamping
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Per-sample normalization (mean=0, std=1)
        # This matches firmware normalization
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db_norm = (mel_spec_db - mean) / (std + 1e-8)
        
        return mel_spec_db_norm


# =============================================================================
# DATASET V2
# =============================================================================

class BirdAudioDatasetV2(Dataset):
    """
    Bird audio dataset with aligned preprocessing and optional augmentations.
    
    Key improvements:
    1. Preprocessing aligned with firmware
    2. Optional audio and spectrogram augmentations
    3. Proper handling of variable-length audio
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        n_mels: int = 40,
        n_fft: int = 512,
        hop_length: int = 256,
        duration: float = 3.0,
        top_db: float = 80.0,
        augment: bool = False,
        audio_augment: Optional[AudioAugment] = None,
        spec_augment: Optional[SpecAugment] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment
        
        # Aligned mel spectrogram transform
        self.mel_transform = AlignedMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            top_db=top_db,
        )
        
        # Augmentations
        self.audio_augment = audio_augment if augment else None
        self.spec_augment = spec_augment if augment else None
        
        # Get class names and create label mapping
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all samples with group information
        self.samples = []
        self.sample_groups = []  # Track which group each sample belongs to
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for audio_file in class_dir.glob("*.wav"):
                self.samples.append((audio_file, self.class_to_idx[class_name]))
                
                # Extract group key for this sample
                name = audio_file.stem
                parts = name.rsplit('_', 1)
                group_key = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
                self.sample_groups.append(group_key)
        
        print(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes")
        for cls in self.classes:
            count = len([s for s in self.samples if s[1] == self.class_to_idx[cls]])
            print(f"  {cls}: {count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.samples[idx]
        
        # Load audio with soundfile (more reliable than torchaudio)
        wav_numpy, sr = sf.read(audio_path)
        waveform = torch.from_numpy(wav_numpy).float()
        
        # Handle shape: soundfile gives (T,) or (T, C), we need (C, T)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or truncate to target length
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
        
        # Apply audio augmentation (before spectrogram)
        if self.audio_augment is not None:
            waveform = self.audio_augment(waveform)
        
        # Extract mel spectrogram (aligned with firmware)
        mel_spec = self.mel_transform(waveform)
        
        # Apply spectrogram augmentation (after spectrogram)
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        
        return mel_spec, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(self.classes)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([class_weights[label] for _, label in self.samples])
        return sample_weights


# =============================================================================
# GROUPED SPLIT FUNCTIONS
# =============================================================================

def get_grouped_indices(
    dataset: BirdAudioDatasetV2,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split dataset indices by recording group to prevent data leakage.
    
    Returns:
        train_indices, val_indices
    """
    # Group samples by source recording
    groups = defaultdict(list)
    for idx, group_key in enumerate(dataset.sample_groups):
        groups[group_key].append(idx)
    
    # Shuffle groups
    group_keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)
    
    # Split at group level
    n_val_groups = int(len(group_keys) * val_split)
    
    val_indices = [idx for key in group_keys[:n_val_groups] for idx in groups[key]]
    train_indices = [idx for key in group_keys[n_val_groups:] for idx in groups[key]]
    
    print(f"\nGrouped split statistics:")
    print(f"  Total groups: {len(group_keys)}")
    print(f"  Training: {len(group_keys) - n_val_groups} groups, {len(train_indices)} samples")
    print(f"  Validation: {n_val_groups} groups, {len(val_indices)} samples")
    print(f"  No clips from same recording in both splits ✓")
    
    return train_indices, val_indices


def create_dataloaders_v2(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    augment_train: bool = True,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Create train and validation dataloaders with GROUPED split.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: DataLoader workers
        seed: Random seed for reproducibility
        augment_train: Whether to apply augmentations to training data
        use_weighted_sampler: Use weighted sampling for class balance
    
    Returns:
        train_loader, val_loader, classes, class_weights
    """
    print("=" * 60)
    print("Creating DataLoaders with Grouped Split (No Leakage)")
    print("=" * 60)
    
    # Create augmentations for training
    audio_aug = AudioAugment(
        noise_snr_range=(15, 30),
        gain_range=(-3, 3),
        p_noise=0.3,
        p_gain=0.5,
    ) if augment_train else None
    
    spec_aug = SpecAugment(
        freq_mask_param=8,  # Up to 8 mel bins masked
        time_mask_param=25,  # Up to 25 time frames masked
        n_freq_masks=2,
        n_time_masks=2,
        p=0.5,
    ) if augment_train else None
    
    # Create datasets
    train_dataset_full = BirdAudioDatasetV2(
        data_dir,
        augment=augment_train,
        audio_augment=audio_aug,
        spec_augment=spec_aug,
    )
    
    val_dataset_full = BirdAudioDatasetV2(
        data_dir,
        augment=False,  # No augmentation for validation
    )
    
    # Get grouped split indices
    train_indices, val_indices = get_grouped_indices(
        train_dataset_full, val_split, seed
    )
    
    # Create subset datasets
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    # Create sampler for training (class balancing)
    if use_weighted_sampler:
        # Get weights for training samples
        full_weights = train_dataset_full.get_sample_weights()
        train_weights = full_weights[train_indices]
        sampler = WeightedRandomSampler(
            train_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        shuffle = False  # Can't use shuffle with sampler
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return (
        train_loader,
        val_loader,
        train_dataset_full.classes,
        train_dataset_full.get_class_weights()
    )


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_no_leakage(
    dataset: BirdAudioDatasetV2,
    train_indices: List[int],
    val_indices: List[int]
) -> bool:
    """Verify that no recording groups appear in both splits."""
    train_groups = set(dataset.sample_groups[i] for i in train_indices)
    val_groups = set(dataset.sample_groups[i] for i in val_indices)
    
    overlap = train_groups & val_groups
    
    if overlap:
        print(f"❌ LEAKAGE DETECTED: {len(overlap)} groups in both splits")
        print(f"   Examples: {list(overlap)[:5]}")
        return False
    else:
        print(f"✓ No leakage: {len(train_groups)} train groups, {len(val_groups)} val groups")
        return True


def get_spectrogram_stats(dataset: BirdAudioDatasetV2, n_samples: int = 100):
    """Get statistics of spectrograms for verification."""
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    all_specs = []
    for idx in indices:
        spec, _ = dataset[idx]
        all_specs.append(spec)
    
    specs = torch.stack(all_specs)
    
    stats = {
        'mean': specs.mean().item(),
        'std': specs.std().item(),
        'min': specs.min().item(),
        'max': specs.max().item(),
        'shape': tuple(specs.shape[1:]),
    }
    
    print(f"\nSpectrogram statistics (n={len(indices)}):")
    print(f"  Shape: {stats['shape']}")
    print(f"  Mean: {stats['mean']:.4f} (expected: ~0)")
    print(f"  Std: {stats['std']:.4f} (expected: ~1)")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    
    return stats


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset v2")
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    args = parser.parse_args()
    
    print("Testing BirdAudioDatasetV2...")
    
    # Create dataset
    dataset = BirdAudioDatasetV2(args.data_dir, augment=False)
    
    # Get split
    train_idx, val_idx = get_grouped_indices(dataset, val_split=0.2)
    
    # Verify no leakage
    verify_no_leakage(dataset, train_idx, val_idx)
    
    # Get stats
    get_spectrogram_stats(dataset)
    
    # Test a sample
    spec, label = dataset[0]
    print(f"\nSample spectrogram shape: {spec.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader, classes, weights = create_dataloaders_v2(
        args.data_dir, batch_size=16
    )
    
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}")
    print(f"Labels: {batch[1]}")
    
    print("\n✓ Dataset v2 working correctly!")
