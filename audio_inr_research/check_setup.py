"""
Setup verification script - Run this first to check everything is ready.
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'torch',
        'torchaudio',
        'numpy',
        'soundfile',
        'tqdm',
        'scipy',
    ]
    
    optional = [
        'pesq',
        'librosa',
        'datasets',
    ]
    
    print("\nChecking required dependencies:")
    all_good = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - NOT INSTALLED")
            all_good = False
    
    print("\nChecking optional dependencies:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"⚠️  {pkg} - Not installed (optional)")
    
    return all_good

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("\n⚠️  CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"\n⚠️  Could not check CUDA: {e}")
        return False

def check_directories():
    """Check if necessary directories exist."""
    print("\nChecking directories:")
    
    data_raw = Path("data/raw")
    data_test = Path("data/test")
    experiments = Path("experiments")
    
    if data_raw.exists():
        audio_files = list(data_raw.glob("*.wav")) + list(data_raw.glob("*.mp3")) + list(data_raw.glob("*.flac"))
        if len(audio_files) > 0:
            print(f"✅ data/raw exists with {len(audio_files)} audio files")
        else:
            print(f"⚠️  data/raw exists but NO audio files found!")
            print("   Please add .wav, .mp3, or .flac files to data/raw/")
    else:
        print(f"❌ data/raw NOT found - creating it")
        data_raw.mkdir(parents=True, exist_ok=True)
        print(f"   Please add audio files to: {data_raw.absolute()}")
    
    if data_test.exists():
        test_files = list(data_test.glob("*.wav")) + list(data_test.glob("*.mp3")) + list(data_test.glob("*.flac"))
        print(f"✅ data/test exists with {len(test_files)} audio files")
    else:
        print(f"⚠️  data/test NOT found - creating it")
        data_test.mkdir(parents=True, exist_ok=True)
        print(f"   Add test files to: {data_test.absolute()}")
    
    if not experiments.exists():
        experiments.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created experiments directory")

def check_model_files():
    """Check if all model files are present."""
    print("\nChecking code files:")
    
    files_to_check = [
        "src/architectures/models.py",
        "src/loss_functions/losses.py",
        "src/metrics/metrics.py",
        "src/utils/coord_utils.py",
        "train.py",
        "evaluate.py",
        "requirements.txt",
    ]
    
    all_good = True
    for file in files_to_check:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_good = False
    
    return all_good

def print_next_steps():
    """Print next steps."""
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("""
1. If any dependencies are missing:
   pip install -r requirements.txt

2. Add audio files to data/raw/:
   - Copy your .wav, .mp3, or .flac files
   - Recommended: 20+ files for training
   - Put 10-20% aside in data/test/ for evaluation

3. Run a quick test (5 epochs, small model):
   python train.py --model siren --data_dir data/raw --epochs 5 --batch_size 4 --hidden_features 128 --num_layers 3

4. If test works, run full training:
   python train.py --model siren --data_dir data/raw --epochs 100
   python train.py --model lisa --data_dir data/raw --epochs 100

5. Evaluate:
   python evaluate.py --checkpoint experiments/.../best_model.pth --test_dir data/test --model siren

For more details, see:
- README.md - Full documentation
- ACTION_PLAN.md - Step-by-step guide
""")

def main():
    print("=" * 50)
    print("AUDIO INR RESEARCH - SETUP VERIFICATION")
    print("=" * 50)
    
    checks = []
    
    checks.append(check_python_version())
    checks.append(check_dependencies())
    check_cuda()
    check_directories()
    checks.append(check_model_files())
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 50)
        print("\nYou're ready to start training! 🚀")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 50)
        print("\nPlease fix the issues above before training.")
    
    print_next_steps()

if __name__ == "__main__":
    main()
