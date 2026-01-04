import torch
import torchaudio
import librosa
import numpy as np
import audiomentations

print("=" * 50)
print("Environment Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Audiomentations version: {audiomentations.__version__}")
print("=" * 50)
print("✓ All dependencies installed correctly!")
