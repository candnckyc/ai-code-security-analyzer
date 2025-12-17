"""
Test script to verify installation and download models
Run this after setup to ensure everything works
"""

import sys
import subprocess

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 50)
    print("AI Code Security Analyzer - Installation Test")
    print("=" * 50)
    print()
    
    # Check Python version
    print("[1/5] Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (Good!)")
    else:
        print(f"✗ Python {version.major}.{version.minor} (Need 3.9+)")
        return False
    print()
    
    # Check essential packages
    print("[2/5] Checking essential packages...")
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "streamlit": "Streamlit",
        "flask": "Flask",
        "pandas": "Pandas",
        "numpy": "NumPy"
    }
    
    all_installed = True
    for package, name in packages.items():
        if check_package(package):
            print(f"✓ {name}")
        else:
            print(f"✗ {name} (Missing!)")
            all_installed = False
    
    if not all_installed:
        print("\nSome packages are missing. Run: pip install -r requirements.txt")
        return False
    print()
    
    # Check CUDA availability
    print("[3/5] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available! GPU: {torch.cuda.get_device_name(0)}")
            print("  Training will be fast!")
        else:
            print("⚠ CUDA not available. Training will use CPU (slower)")
            print("  This is OK for development!")
    except Exception as e:
        print(f"⚠ Could not check CUDA: {e}")
    print()
    
    # Test model download
    print("[4/5] Testing model download...")
    try:
        from transformers import AutoTokenizer, AutoModel
        print("  Downloading CodeBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        print("  ✓ Tokenizer downloaded!")
        
        # Test tokenization
        code = "def hello(): return 'world'"
        tokens = tokenizer(code, return_tensors="pt")
        print(f"  ✓ Tokenizer works! Generated {tokens['input_ids'].shape[1]} tokens")
        
        print("\n  Note: Full model will download on first use (~500MB)")
        print("  This is normal and only happens once.")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    print()
    
    # Check directory structure
    print("[5/5] Checking project structure...")
    import os
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "models/pretrained",
        "models/finetuned",
        "src/preprocessing",
        "src/model",
        "src/analysis",
        "src/api",
        "src/ui",
        "notebooks",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (Missing!)")
            all_exist = False
    
    if not all_exist:
        print("\nSome directories are missing. Run the setup script again.")
        return False
    print()
    
    # Success!
    print("=" * 50)
    print("✓ ALL CHECKS PASSED!")
    print("=" * 50)
    print("\nYou're ready to start development!")
    print("\nNext steps:")
    print("1. Run: python download_dataset.py")
    print("2. Open: notebooks/01_data_exploration.ipynb")
    print("3. Start coding!")
    print()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
