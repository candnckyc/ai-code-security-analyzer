@echo off
REM Windows Setup Script for AI Code Security Analyzer
REM Run this script to set up the entire project automatically

echo ========================================
echo AI Code Security Analyzer - Setup
echo ========================================
echo.

REM Check Python installation
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python found!
echo.

REM Create project structure
echo [2/8] Creating project structure...
mkdir data\raw data\processed data\samples 2>nul
mkdir models\pretrained models\finetuned models\configs 2>nul
mkdir src\preprocessing src\model src\analysis src\api src\ui 2>nul
mkdir notebooks tests docs logs 2>nul
echo Project directories created!
echo.

REM Create __init__.py files
echo [3/8] Creating Python package files...
type nul > src\__init__.py
type nul > src\preprocessing\__init__.py
type nul > src\model\__init__.py
type nul > src\analysis\__init__.py
type nul > src\api\__init__.py
type nul > src\ui\__init__.py
echo Package files created!
echo.

REM Create .gitkeep files
echo [4/8] Creating .gitkeep files...
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > models\pretrained\.gitkeep
type nul > models\finetuned\.gitkeep
type nul > logs\.gitkeep
echo .gitkeep files created!
echo.

REM Create virtual environment
echo [5/8] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created!
echo.

REM Activate virtual environment and install dependencies
echo [6/8] Installing dependencies...
echo This may take 5-10 minutes...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Dependencies installed!
echo.

REM Download model
echo [7/8] Testing model download...
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base'); print('Model tokenizer downloaded successfully!')"
if errorlevel 1 (
    echo WARNING: Model download failed. You can download it later.
) else (
    echo Model components downloaded!
)
echo.

REM Final instructions
echo [8/8] Setup complete!
echo.
echo ========================================
echo NEXT STEPS:
echo ========================================
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Download dataset:
echo    python download_dataset.py
echo.
echo 3. Start development!
echo.
echo Repository structure is ready for Git commit.
echo ========================================
pause
