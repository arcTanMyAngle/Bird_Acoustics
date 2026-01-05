# ~/bird-detection/scripts/dataset.py

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf

class BirdAudioDataset(Dataset):
    """Dataset for bird audio classification."""
    
    def __init__(self, data_dir, sample_rate=16000, n_mels=40, 
                 n_fft=512, hop_length=256, duration=3.0):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Amplitude to dB
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80)
        
        # Get class names and create label mapping
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Collect all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for audio_file in class_dir.glob("*.wav"):
                self.samples.append((audio_file, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples across {len(self.classes)} classes")
        for cls in self.classes:
            count = len([s for s in self.samples if s[1] == self.class_to_idx[cls]])
            print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load audio
        # Load with soundfile to bypass torchaudio backend issues
        wav_numpy, sr = sf.read(audio_path)
        waveform = torch.from_numpy(wav_numpy).float()

        # Handle shape (Soundfile gives Time x Channels, we need Channels x Time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # Add channel dim for mono
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
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        class_counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.classes)
        
        return torch.FloatTensor(weights)


def create_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """Create train and validation dataloaders."""
    
    dataset = BirdAudioDataset(data_dir)
    
    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.classes, dataset.get_class_weights()


if __name__ == "__main__":
    # Test the dataset
    dataset = BirdAudioDataset("data/augmented")
    sample, label = dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")
