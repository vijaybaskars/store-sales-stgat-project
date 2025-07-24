#!/usr/bin/env python3
"""
GPU Detection and PyTorch Installation Recommendations
Helps users choose the optimal PyTorch installation for their system
"""

import os
import platform
import subprocess
import sys

def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is available"""
    try:
        # Try nvidia-smi command
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def detect_cuda():
    """Detect if CUDA is installed"""
    try:
        # Check for CUDA installation
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            # Extract CUDA version
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    return line.strip()
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def detect_metal_gpu():
    """Detect if Apple Metal GPU is available (Mac M1/M2)"""
    if platform.system() != 'Darwin':
        return False
    
    try:
        # Check for Apple Silicon
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            output = result.stdout.lower()
            return 'apple' in output and ('m1' in output or 'm2' in output or 'm3' in output)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return False

def get_installation_recommendation():
    """Get PyTorch installation recommendation based on hardware"""
    
    print("🔍 GPU Detection and PyTorch Installation Recommendations")
    print("=" * 60)
    
    # System info
    print(f"💻 System: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # NVIDIA GPU detection
    has_nvidia = detect_nvidia_gpu()
    cuda_version = detect_cuda()
    has_metal = detect_metal_gpu()
    
    print(f"\n🎮 Hardware Detection:")
    print(f"   NVIDIA GPU: {'✅ Found' if has_nvidia else '❌ Not found'}")
    print(f"   CUDA: {'✅ ' + str(cuda_version) if cuda_version else '❌ Not found'}")
    print(f"   Apple Metal: {'✅ Found' if has_metal else '❌ Not found'}")
    
    print(f"\n📦 Installation Recommendations:")
    
    if has_nvidia and cuda_version:
        print("🚀 NVIDIA GPU with CUDA detected - GPU acceleration recommended!")
        print("\n   Install GPU-accelerated PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   OR use: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("\n   Then install other neural dependencies:")
        print("   pip install torch-geometric pytorch-lightning lightning-utilities torchmetrics")
        
    elif has_metal:
        print("🍎 Apple Silicon detected - Metal acceleration available!")
        print("\n   Install MPS-accelerated PyTorch:")
        print("   pip install torch torchvision torchaudio")
        print("   pip install torch-geometric pytorch-lightning lightning-utilities torchmetrics")
        print("\n   Note: PyTorch will automatically use Metal Performance Shaders (MPS)")
        
    else:
        print("💻 CPU-only system detected - Using CPU-optimized PyTorch")
        print("\n   Install CPU-only PyTorch (smaller download, faster for CPU):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("   OR use: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
        print("\n   Then install other neural dependencies:")
        print("   pip install torch-geometric pytorch-lightning lightning-utilities torchmetrics")
    
    print(f"\n💡 For this project:")
    if has_nvidia or has_metal:
        print("   • Neural models will run faster with GPU acceleration")
        print("   • You can enable neural models: set enable_neural_models = True in src/serving/config.py")
        print("   • GPU acceleration is most beneficial for training new models")
    else:
        print("   • Neural models will run on CPU (slower but functional)")
        print("   • Consider keeping neural models disabled for production (current default)")
        print("   • Traditional models work great on CPU and are currently the primary models")
    
    print(f"\n🔧 Quick Test Commands:")
    print("   python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"")
    if has_nvidia:
        print("   python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
    if has_metal:
        print("   python -c \"import torch; print(f'MPS available: {torch.backends.mps.is_available()}')\"")
    
    print(f"\n📋 Summary:")
    if has_nvidia and cuda_version:
        return "GPU_NVIDIA"
    elif has_metal:
        return "GPU_METAL"  
    else:
        return "CPU_ONLY"

if __name__ == "__main__":
    recommendation = get_installation_recommendation()
    
    print(f"\n🎯 Recommendation: {recommendation}")
    print("\nRun this script anytime to check your hardware configuration!")