#!/usr/bin/env python3
"""scripts/verify_environment.py - Level 1 Boss Fight"""

import sys
import importlib.metadata
from pathlib import Path

REQUIRED_VERSIONS = {
    "torch": "2.6.0",
    "torchaudio": "2.6.0", 
    "ai-edge-torch": "0.7.0",
    "ai-edge-litert": "2.0.3",
    "librosa": "0.10.2",
    "numpy": "1.",  # prefix match for 1.x
}

def verify_environment():
    print("=" * 60)
    print("🎮 LEVEL 1 BOSS FIGHT: Environment Verification")
    print("=" * 60)
    
    failures = []
    
    for package, expected in REQUIRED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
            if not actual.startswith(expected.rstrip('.')):
                failures.append(f"❌ {package}: expected {expected}*, got {actual}")
            else:
                print(f"✅ {package}: {actual}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"❌ {package}: NOT INSTALLED")
    
    # Verify torch can export
    print("\n🔬 Testing torch.export capability...")
    try:
        import torch
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x * 2
        
        model = DummyModel()
        exported = torch.export.export(model, (torch.randn(1, 10),))
        print("✅ torch.export: functional")
    except Exception as e:
        failures.append(f"❌ torch.export failed: {e}")
    
    # Verify ai-edge-torch import
    print("\n🔬 Testing ai-edge-torch import...")
    try:
        import ai_edge_torch
        print(f"✅ ai-edge-torch: imported successfully")
    except Exception as e:
        failures.append(f"❌ ai-edge-torch import failed: {e}")
    
    print("\n" + "=" * 60)
    if failures:
        print("💀 BOSS FIGHT FAILED")
        for f in failures:
            print(f"   {f}")
        print("\n🔧 Fix: Run 'uv sync --frozen' in Docker container")
        sys.exit(1)
    else:
        print("🏆 BOSS FIGHT PASSED - Level 1 Complete!")
        print("=" * 60)

if __name__ == "__main__":
    verify_environment()