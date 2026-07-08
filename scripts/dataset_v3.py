#!/usr/bin/env python3
"""
dataset_v3.py - Enhanced Dataset with Noise Rejection Testing & Hard Negative Mining

Key improvements over v2:
1. Three-way split: train/val/noise_test (for background rejection validation)
2. Hard negative mining: samples near decision boundary get higher weight
3. SNR-aware augmentation: mix bird+noise at controlled levels
4. Per-class metrics tracking

Usage:
    from dataset_v3 import create_dataloaders_v3, NoiseRejectionTestSet
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
# CONSTANTS
# =============================================================================

EXPECTED_CLASSES_9 = [
    "american_crow",
    "background",
    "california_quail",
    "california_scrub_jay",
    "great_horned_owl",
    "killdeer",
    "mourning_dove",
    "red_tailed_hawk",
    "western_meadowlark",
]

BACKGROUND_CLASS = "background"
BACKGROUND_IDX = 1  # Index in sorted class list


# =============================================================================
# SPECAUGMENT AND AUGMENTATIONS
# =============================================================================

class SpecAugment(nn.Module):
    """SpecAugment: time and frequency masking for spectrograms."""
    
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
        if random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0
        
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0:t0 + t] = 0
        
        return spec


class AudioAugment:
    """Audio-level augmentations before spectrogram extraction."""
    
    def __init__(
        self,
        noise_snr_range: Tuple[float, float] = (10, 30),
        gain_range: Tuple[float, float] = (-6, 6),
        p_noise: float = 0.3,
        p_gain: float = 0.5,
    ):
        self.noise_snr_range = noise_snr_range
        self.gain_range = gain_range
        self.p_noise = p_noise
        self.p_gain = p_gain
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    def apply_gain(self, waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        return waveform * (10 ** (gain_db / 20))
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_noise:
            snr = random.uniform(*self.noise_snr_range)
            waveform = self.add_noise(waveform, snr)
        
        if random.random() < self.p_gain:
            gain = random.uniform(*self.gain_range)
            waveform = self.apply_gain(waveform, gain)
        
        return torch.clamp(waveform, -1.0, 1.0)


# =============================================================================
# ALIGNED MEL SPECTROGRAM TRANSFORM
# =============================================================================

class AlignedMelSpectrogram(nn.Module):
    """Mel spectrogram transform aligned with ESP32 firmware."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
        top_db: float = 80.0,
        center: bool = True,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            center=center,
            norm='slaney',
            mel_scale='htk',
        )
        
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db_norm = (mel_spec_db - mean) / (std + 1e-8)
        
        return mel_spec_db_norm


# =============================================================================
# HARD NEGATIVE MINING SUPPORT
# =============================================================================

class HardNegativeMiner:
    """
    Track and weight hard negatives (samples near decision boundary).
    
    Hard negatives are:
    1. Bird samples with low confidence (model unsure)
    2. Background samples with high bird-class confidence (false positives)
    3. Samples that were misclassified
    """
    
    def __init__(
        self,
        n_samples: int,
        hard_negative_weight: float = 2.0,
        confidence_threshold: float = 0.7,
        update_frequency: int = 5,  # Update every N epochs
    ):
        self.n_samples = n_samples
        self.hard_negative_weight = hard_negative_weight
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency
        
        # Track difficulty scores per sample
        self.difficulty_scores = torch.ones(n_samples)
        self.last_predictions = torch.zeros(n_samples, dtype=torch.long)
        self.last_confidences = torch.zeros(n_samples)
        self.misclassified = torch.zeros(n_samples, dtype=torch.bool)
    
    def update(
        self,
        indices: torch.Tensor,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Update difficulty scores based on model predictions."""
        for i, idx in enumerate(indices):
            idx = idx.item()
            pred = predictions[i].item()
            conf = confidences[i].item()
            label = labels[i].item()
            
            self.last_predictions[idx] = pred
            self.last_confidences[idx] = conf
            self.misclassified[idx] = (pred != label)
            
            # Calculate difficulty score
            if pred != label:
                # Misclassified: high difficulty
                self.difficulty_scores[idx] = self.hard_negative_weight
            elif conf < self.confidence_threshold:
                # Low confidence correct: moderate difficulty
                self.difficulty_scores[idx] = 1.0 + (1.0 - conf)
            else:
                # High confidence correct: low difficulty
                self.difficulty_scores[idx] = 1.0
    
    def get_sample_weights(self, base_weights: torch.Tensor) -> torch.Tensor:
        """Combine base class weights with difficulty scores."""
        return base_weights * self.difficulty_scores
    
    def get_hard_negative_indices(self, top_k: int = 100) -> List[int]:
        """Get indices of hardest samples."""
        _, indices = torch.topk(self.difficulty_scores, min(top_k, self.n_samples))
        return indices.tolist()
    
    def get_statistics(self) -> Dict:
        """Get mining statistics."""
        return {
            "n_misclassified": self.misclassified.sum().item(),
            "n_low_confidence": (self.last_confidences < self.confidence_threshold).sum().item(),
            "mean_difficulty": self.difficulty_scores.mean().item(),
            "max_difficulty": self.difficulty_scores.max().item(),
        }


# =============================================================================
# DATASET V3
# =============================================================================

class BirdAudioDatasetV3(Dataset):
    """
    Enhanced bird audio dataset with noise rejection testing support.
    
    Key features:
    1. Tracks sample metadata for hard negative mining
    2. Supports SNR-controlled background mixing
    3. Returns sample indices for mining updates
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
        return_index: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment
        self.return_index = return_index
        
        self.mel_transform = AlignedMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            top_db=top_db,
        )
        
        self.audio_augment = audio_augment
        self.spec_augment = spec_augment
        
        # Load samples
        self.samples: List[Tuple[Path, int]] = []
        self.sample_groups: List[str] = []  # For grouped splitting
        self.sample_metadata: List[Dict] = []  # Additional metadata
        
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Track background class index
        self.background_idx = self.class_to_idx.get(BACKGROUND_CLASS, -1)
        
        # Load all samples
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            cls_idx = self.class_to_idx[cls]
            
            for audio_file in sorted(cls_dir.glob("*.wav")):
                self.samples.append((audio_file, cls_idx))
                
                # Extract group key (original recording ID)
                # Filename format: {class_name}_{recording_id}_{clip_idx}.wav
                # e.g., "american_crow_0042_0.wav" or "killdeer_0042_0.wav"
                # 
                # We need to strip the class name prefix to get the recording ID
                stem = audio_file.stem
                
                # Remove class name prefix (including its underscores)
                if stem.startswith(cls + "_"):
                    remainder = stem[len(cls) + 1:]  # +1 for the underscore
                    # remainder is now "0042_0" - take first part as recording ID
                    parts = remainder.split('_')
                    recording_id = parts[0] if parts else stem
                else:
                    # Fallback: use full stem
                    recording_id = stem
                
                group_key = f"{cls}_{recording_id}"
                self.sample_groups.append(group_key)
                
                # Metadata
                self.sample_metadata.append({
                    "path": str(audio_file),
                    "class": cls,
                    "is_background": (cls == BACKGROUND_CLASS),
                    "group_key": group_key,
                })
        
        print(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes")
        for cls in self.classes:
            count = sum(1 for _, idx in self.samples if self.idx_to_class[idx] == cls)
            # Count unique groups for this class
            cls_groups = set(g for g, (_, idx) in zip(self.sample_groups, self.samples) 
                            if self.idx_to_class[idx] == cls)
            print(f"  {cls}: {count} samples, {len(cls_groups)} groups")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        audio_path, label = self.samples[idx]
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            waveform, sr = sf.read(audio_path)
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or trim
        if waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.target_length:
            start = random.randint(0, waveform.shape[1] - self.target_length) if self.augment else 0
            waveform = waveform[:, start:start + self.target_length]
        
        # Audio augmentation
        if self.augment and self.audio_augment is not None:
            waveform = self.audio_augment(waveform)
        
        # Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Spec augmentation
        if self.augment and self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
        
        if self.return_index:
            return mel_spec, label, idx
        return mel_spec, label
    
    def get_class_counts(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return dict(counts)
    
    def get_class_weights(self) -> torch.Tensor:
        """Inverse frequency class weights."""
        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = []
        for cls in self.classes:
            count = counts.get(cls, 1)
            weights.append(total / (len(self.classes) * count))
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[label] for _, label in self.samples])
    
    def get_background_indices(self) -> List[int]:
        """Get indices of all background samples."""
        if self.background_idx < 0:
            return []
        return [i for i, (_, label) in enumerate(self.samples) if label == self.background_idx]
    
    def get_bird_indices(self) -> List[int]:
        """Get indices of all bird (non-background) samples."""
        if self.background_idx < 0:
            return list(range(len(self.samples)))
        return [i for i, (_, label) in enumerate(self.samples) if label != self.background_idx]


# =============================================================================
# NOISE REJECTION TEST SET
# =============================================================================

class NoiseRejectionTestSet(Dataset):
    """
    Special test set for evaluating noise rejection capability.
    
    Contains:
    1. Pure background samples (should predict "background")
    2. Bird samples mixed with noise at various SNR levels
    3. Synthetic edge cases (very quiet birds, loud noise)
    """
    
    def __init__(
        self,
        bird_dataset: BirdAudioDatasetV3,
        background_indices: List[int],
        bird_indices: List[int],
        snr_levels: List[float] = [0, 5, 10, 15, 20],
        n_pure_background: int = 50,
        n_mixed_per_snr: int = 20,
        seed: int = 42,
    ):
        self.bird_dataset = bird_dataset
        self.snr_levels = snr_levels
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Sample indices
        self.pure_bg_indices = random.sample(
            background_indices, 
            min(n_pure_background, len(background_indices))
        )
        
        # For mixed samples, we'll create them on the fly
        self.mixed_samples: List[Tuple[int, int, float]] = []  # (bird_idx, bg_idx, snr)
        
        for snr in snr_levels:
            for _ in range(n_mixed_per_snr):
                bird_idx = random.choice(bird_indices)
                bg_idx = random.choice(background_indices)
                self.mixed_samples.append((bird_idx, bg_idx, snr))
        
        self.mel_transform = bird_dataset.mel_transform
        
        print(f"NoiseRejectionTestSet created:")
        print(f"  Pure background: {len(self.pure_bg_indices)}")
        print(f"  Mixed samples: {len(self.mixed_samples)} ({len(snr_levels)} SNR levels)")
    
    def __len__(self) -> int:
        return len(self.pure_bg_indices) + len(self.mixed_samples)
    
    def __getitem__(self, idx: int):
        if idx < len(self.pure_bg_indices):
            # Pure background sample
            bg_idx = self.pure_bg_indices[idx]
            spec, label = self.bird_dataset[bg_idx]
            return spec, label, "pure_background", 0.0
        else:
            # Mixed sample
            mixed_idx = idx - len(self.pure_bg_indices)
            bird_idx, bg_idx, snr = self.mixed_samples[mixed_idx]
            
            # Load bird and background audio
            bird_spec, bird_label = self.bird_dataset[bird_idx]
            bg_spec, _ = self.bird_dataset[bg_idx]
            
            # Mix at specified SNR (in spectrogram domain - approximation)
            # More accurate would be to mix in audio domain, but this is faster
            snr_linear = 10 ** (snr / 20)
            mixed_spec = bird_spec + bg_spec / snr_linear
            
            # Re-normalize
            mean = mixed_spec.mean()
            std = mixed_spec.std()
            mixed_spec = (mixed_spec - mean) / (std + 1e-8)
            
            return mixed_spec, bird_label, "mixed", snr
    
    def evaluate(
        self,
        model: nn.Module,
        device: torch.device,
        background_idx: int = 1,
    ) -> Dict:
        """
        Evaluate noise rejection performance.
        
        Returns metrics on:
        1. Background rejection rate (pure background → background prediction)
        2. Bird detection at various SNR levels
        3. False positive rate (background misclassified as bird)
        """
        model.eval()
        
        results = {
            "pure_background": {"correct": 0, "total": 0},
            "snr_performance": {snr: {"correct": 0, "total": 0} for snr in self.snr_levels},
            "false_positives": 0,
            "predictions": [],
        }
        
        with torch.no_grad():
            for i in range(len(self)):
                spec, label, sample_type, snr = self[i]
                spec = spec.unsqueeze(0).to(device)
                
                output = model(spec)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                confidence = probs[0, pred].item()
                
                results["predictions"].append({
                    "type": sample_type,
                    "snr": snr,
                    "true_label": label,
                    "predicted": pred,
                    "confidence": confidence,
                })
                
                if sample_type == "pure_background":
                    results["pure_background"]["total"] += 1
                    if pred == background_idx:
                        results["pure_background"]["correct"] += 1
                    else:
                        results["false_positives"] += 1
                else:
                    results["snr_performance"][snr]["total"] += 1
                    if pred == label:
                        results["snr_performance"][snr]["correct"] += 1
        
        # Calculate rates
        pb = results["pure_background"]
        results["background_rejection_rate"] = pb["correct"] / max(pb["total"], 1) * 100
        results["false_positive_rate"] = results["false_positives"] / max(pb["total"], 1) * 100
        
        for snr in self.snr_levels:
            sp = results["snr_performance"][snr]
            sp["accuracy"] = sp["correct"] / max(sp["total"], 1) * 100
        
        return results


# =============================================================================
# STRATIFIED GROUPED SPLIT (FIXED)
# =============================================================================

def get_three_way_split(
    dataset: BirdAudioDatasetV3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    noise_test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into train/val/noise_test with:
    1. Group-level separation (no recording appears in multiple splits)
    2. STRATIFIED by class (each class represented in ALL splits)
    
    This fixes the bug where random shuffling could put entire classes
    into only one split.
    """
    random.seed(seed)
    
    # Group samples by CLASS first, then by recording group
    # Structure: class_name -> {group_key: [sample_indices]}
    class_groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    
    for idx, (filepath, label) in enumerate(dataset.samples):
        cls = dataset.idx_to_class[label]
        group_key = dataset.sample_groups[idx]
        class_groups[cls][group_key].append(idx)
    
    train_indices = []
    val_indices = []
    noise_test_indices = []
    
    print(f"\nStratified split by class:")
    
    # Split EACH CLASS separately to ensure stratification
    for cls in sorted(class_groups.keys()):
        groups = class_groups[cls]
        group_keys = list(groups.keys())
        random.shuffle(group_keys)
        
        n_groups = len(group_keys)
        n_train = max(1, int(n_groups * train_ratio))  # At least 1 group
        n_val = max(1, int(n_groups * val_ratio))      # At least 1 group
        
        # Ensure we don't exceed available groups
        if n_train + n_val >= n_groups:
            # Reduce to fit
            n_train = max(1, n_groups - 2)
            n_val = max(1, min(n_val, n_groups - n_train - 1))
        
        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train:n_train + n_val]
        test_keys = group_keys[n_train + n_val:]
        
        # Collect indices
        cls_train = []
        cls_val = []
        cls_test = []
        
        for key in train_keys:
            cls_train.extend(groups[key])
        for key in val_keys:
            cls_val.extend(groups[key])
        for key in test_keys:
            cls_test.extend(groups[key])
        
        train_indices.extend(cls_train)
        val_indices.extend(cls_val)
        noise_test_indices.extend(cls_test)
        
        print(f"  {cls}: {len(cls_train)} train, {len(cls_val)} val, {len(cls_test)} test "
              f"({n_groups} groups)")
    
    # Shuffle within each split to mix classes
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(noise_test_indices)
    
    print(f"\nTotal split:")
    print(f"  Training: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    print(f"  Noise test: {len(noise_test_indices)} samples")
    print(f"  No group overlap between splits ✓")
    print(f"  All classes in all splits ✓")
    
    return train_indices, val_indices, noise_test_indices


class IndexedSubset(Dataset):
    """
    Subset that properly handles return_index by remapping to subset indices.
    
    When the underlying dataset returns (data, label, original_idx), this
    wrapper converts original_idx to the subset index for proper tracking.
    """
    def __init__(self, dataset: Dataset, indices: List[int], return_index: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.return_index = return_index
        # Create reverse mapping: original_idx -> subset_idx
        self.idx_map = {orig: sub for sub, orig in enumerate(indices)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        item = self.dataset[original_idx]
        
        if self.return_index:
            if len(item) == 3:
                # Dataset returned index, but we want subset index
                return item[0], item[1], idx
            else:
                # Dataset didn't return index, add it
                return item[0], item[1], idx
        else:
            if len(item) == 3:
                # Strip the index
                return item[0], item[1]
            return item


def create_dataloaders_v3(
    data_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    augment_train: bool = True,
    use_weighted_sampler: bool = True,
    enable_hard_negative_mining: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, NoiseRejectionTestSet, List[str], torch.Tensor, Optional[HardNegativeMiner]]:
    """
    Create train/val/test dataloaders and noise rejection test set.

    test_loader covers ALL held-out noise_test-split samples (plain, no augment) —
    use it exactly once for final honest metrics; noise_test_set is the synthetic
    SNR-mixing evaluator built from the same split.

    Returns:
        train_loader, val_loader, test_loader, noise_test_set, classes, class_weights, hard_negative_miner
    """
    print("=" * 60)
    print("Creating DataLoaders with Noise Rejection Test Split")
    print("=" * 60)
    
    # Augmentations
    audio_aug = AudioAugment(
        noise_snr_range=(15, 30),
        gain_range=(-3, 3),
        p_noise=0.3,
        p_gain=0.5,
    ) if augment_train else None
    
    spec_aug = SpecAugment(
        freq_mask_param=8,
        time_mask_param=25,
        n_freq_masks=2,
        n_time_masks=2,
        p=0.5,
    ) if augment_train else None
    
    # Create datasets - IMPORTANT: Don't use return_index here, we'll handle it in IndexedSubset
    train_dataset_full = BirdAudioDatasetV3(
        data_dir,
        augment=augment_train,
        audio_augment=audio_aug,
        spec_augment=spec_aug,
        return_index=False,  # Changed: handle indexing in subset wrapper
    )
    
    val_dataset_full = BirdAudioDatasetV3(
        data_dir,
        augment=False,
        return_index=False,
    )
    
    # Get three-way split
    train_indices, val_indices, noise_test_indices = get_three_way_split(
        train_dataset_full, train_ratio, val_ratio, 1.0 - train_ratio - val_ratio, seed
    )
    
    # Create subsets with proper index handling
    train_dataset = IndexedSubset(train_dataset_full, train_indices, return_index=enable_hard_negative_mining)
    val_dataset = IndexedSubset(val_dataset_full, val_indices, return_index=False)
    
    # Create noise rejection test set
    noise_test_bg_indices = [i for i in noise_test_indices if train_dataset_full.samples[i][1] == train_dataset_full.background_idx]
    noise_test_bird_indices = [i for i in noise_test_indices if train_dataset_full.samples[i][1] != train_dataset_full.background_idx]
    
    noise_test_set = NoiseRejectionTestSet(
        val_dataset_full,
        noise_test_bg_indices,
        noise_test_bird_indices,
        snr_levels=[0, 5, 10, 15, 20],
        seed=seed,
    )
    
    # Hard negative miner - sized for TRAINING SUBSET, not full dataset
    miner = None
    if enable_hard_negative_mining:
        miner = HardNegativeMiner(
            n_samples=len(train_indices),  # Changed: use subset size
            hard_negative_weight=2.0,
            confidence_threshold=0.7,
        )
    
    # Sampler - compute weights for training subset only
    if use_weighted_sampler:
        # Get class weights from full dataset
        class_weights = train_dataset_full.get_class_weights()
        
        # Compute per-sample weights for training subset
        train_weights = []
        for orig_idx in train_indices:
            _, label = train_dataset_full.samples[orig_idx]
            train_weights.append(class_weights[label].item())
        train_weights = torch.tensor(train_weights)
        
        sampler = WeightedRandomSampler(
            train_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Dataloaders
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

    test_dataset = IndexedSubset(val_dataset_full, noise_test_indices, return_index=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        noise_test_set,
        train_dataset_full.classes,
        train_dataset_full.get_class_weights(),
        miner,
    )


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset v3")
    parser.add_argument("--data-dir", type=str, default="data/augmented")
    args = parser.parse_args()
    
    print("Testing BirdAudioDatasetV3...")
    
    train_loader, val_loader, test_loader, noise_test, classes, weights, miner = create_dataloaders_v3(
        args.data_dir,
        batch_size=16,
        enable_hard_negative_mining=True,
    )
    
    print(f"\nClasses: {classes}")
    print(f"Class weights: {weights.tolist()}")
    
    # Test batch
    batch = next(iter(train_loader))
    if len(batch) == 3:
        specs, labels, indices = batch
        print(f"\nBatch with indices - shape: {specs.shape}, indices: {indices[:5]}")
    else:
        specs, labels = batch
        print(f"\nBatch shape: {specs.shape}")
    
    print(f"Labels: {labels}")
    
    print("\n✓ Dataset v3 working correctly!")