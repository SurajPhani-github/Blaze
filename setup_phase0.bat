@echo off
REM CurricuRL Phase 0 Setup Script (Windows)
REM =========================================

echo ==============================================
echo   CurricuRL Phase 0 Setup
echo ==============================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.8+
    exit /b 1
)
echo Python OK
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Install dependencies
echo Installing Phase 0 dependencies...
python -m pip install --upgrade pip
pip install -r requirements_phase0.txt
echo Dependencies installed
echo.

REM Download spaCy model
echo Downloading spaCy language model...
python -m spacy download en_core_web_sm
echo spaCy model downloaded
echo.

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
echo NLTK data downloaded
echo.

REM Check input files
echo Checking input files...
if not exist "coursera_courses.csv" (
    echo WARNING: coursera_courses.csv not found!
    echo    Please ensure this file is in the current directory
) else (
    echo coursera_courses.csv found
)

if not exist "augmented_learner_data.csv" (
    echo WARNING: augmented_learner_data.csv not found!
    echo    Please ensure this file is in the current directory
) else (
    echo augmented_learner_data.csv found
)
echo.

REM Create directories
echo Creating output directories...
if not exist "data" mkdir data
if not exist "data\neo4j" mkdir data\neo4j
if not exist "outputs" mkdir outputs
if not exist "models" mkdir models
echo Directories created
echo.

echo ==============================================
echo   Setup Complete!
echo ==============================================
echo.
echo Next steps:
echo   1. Ensure coursera_courses.csv and augmented_learner_data.csv are present
echo   2. Run: venv\Scripts\activate.bat (to activate virtual env)
echo   3. Run: python run_phase0.py (to execute pipeline)
echo.
echo For help, see README_PHASE0.md
echo.
pause