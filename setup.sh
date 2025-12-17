#!/bin/bash
# Mac/Linux Setup Script for AI Code Security Analyzer
# Run: chmod +x setup.sh && ./setup.sh

echo "========================================"
echo "AI Code Security Analyzer - Setup"
echo "========================================"
echo ""

# Check Python installation
echo "[1/8] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Install Python from https://www.python.org/downloads/"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"
echo ""

# Create project structure
echo "[2/8] Creating project structure..."
mkdir -p data/{raw,processed,samples}
mkdir -p models/{pretrained,finetuned,configs}
mkdir -p src/{preprocessing,model,analysis,api,ui}
mkdir -p notebooks tests docs logs
echo "✓ Project directories created!"
echo ""

# Create __init__.py files
echo "[3/8] Creating Python package files..."
touch src/__init__.py
touch src/{preprocessing,model,analysis,api,ui}/__init__.py
echo "✓ Package files created!"
echo ""

# Create .gitkeep files
echo "[4/8] Creating .gitkeep files..."
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/pretrained/.gitkeep
touch models/finetuned/.gitkeep
touch logs/.gitkeep
echo "✓ .gitkeep files created!"
echo ""

# Create virtual environment
echo "[5/8] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi
echo "✓ Virtual environment created!"
echo ""

# Activate virtual environment and install dependencies
echo "[6/8] Installing dependencies..."
echo "This may take 5-10 minutes..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi
echo "✓ Dependencies installed!"
echo ""

# Download model
echo "[7/8] Testing model download..."
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base'); print('✓ Model tokenizer downloaded successfully!')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ Model download will happen on first run"
else
    echo "✓ Model components downloaded!"
fi
echo ""

# Final instructions
echo "[8/8] Setup complete! ✓"
echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download dataset:"
echo "   python download_dataset.py"
echo ""
echo "3. Start development!"
echo ""
echo "Repository structure is ready for Git commit."
echo "========================================"
