@echo off
setlocal enabledelayedexpansion

:: ANSI color codes for Windows
set "RED=[91m"
set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "BOLD=[1m"
set "NC=[0m"

:: Print with color and formatting
call :log "Setting up OCEAN..."

:: Check prerequisites
call :check_prerequisites
if errorlevel 1 exit /b 1

:: Install Git LFS
call :install_git_lfs
if errorlevel 1 exit /b 1

:: Setup Python environment
call :setup_python_env
if errorlevel 1 exit /b 1

:: Setup Hugging Face
call :setup_huggingface
if errorlevel 1 exit /b 1

:: Download datasets
call :download_datasets
if errorlevel 1 exit /b 1

:: Setup complete
echo.
call :success "Setup Complete!"
echo.
echo To get started:
echo 1. Activate the virtual environment:
echo    %YELLOW%.venv\Scripts\activate%NC%
echo.
echo 2. Run experiments:
echo    %YELLOW%python scripts\run_experiments.py%NC%
echo.
echo For more information, please refer to the README.md file.
exit /b 0

:check_prerequisites
call :log "Checking prerequisites..."

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    call :error "Python is not installed"
    exit /b 1
)
python -c "import sys; assert sys.version_info >= (3,8)" >nul 2>&1
if errorlevel 1 (
    call :error "Python 3.8+ is required"
    exit /b 1
)
call :success "Python 3.8+ is installed"

:: Check Git
git --version >nul 2>&1
if errorlevel 1 (
    call :error "Git is not installed"
    exit /b 1
)
call :success "Git is installed"

:: Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    call :error "pip is not installed"
    exit /b 1
)
call :success "pip is installed"
exit /b 0

:install_git_lfs
call :log "Installing Git LFS..."

:: Check if Git LFS is already installed
git lfs --version >nul 2>&1
if not errorlevel 1 (
    call :success "Git LFS is already installed"
    exit /b 0
)

:: Install Git LFS
call :warn "Please install Git LFS manually from: https://git-lfs.com"
call :warn "After installation, run: git lfs install"
exit /b 1

:setup_python_env
call :log "Setting up Python environment..."

:: Create virtual environment
python -m venv .venv
if errorlevel 1 (
    call :error "Failed to create virtual environment"
    exit /b 1
)
call :success "Created virtual environment"

:: Install dependencies
call .venv\Scripts\activate.bat
if errorlevel 1 (
    call :error "Failed to activate virtual environment"
    exit /b 1
)
call :success "Activated virtual environment"

python -m pip install --upgrade pip
if errorlevel 1 (
    call :error "Failed to upgrade pip"
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    call :error "Failed to install dependencies"
    exit /b 1
)
call :success "Installed dependencies"
exit /b 0

:setup_huggingface
call :log "Setting up Hugging Face..."

where huggingface-cli >nul 2>&1
if errorlevel 1 (
    call :warn "huggingface-cli not found. Installing..."
    pip install --upgrade huggingface_hub[cli]
)

call :warn "Please follow the instructions to log in to Hugging Face:"
huggingface-cli login
if errorlevel 1 (
    call :error "Failed to login to Hugging Face"
    exit /b 1
)
call :success "Hugging Face setup complete"
exit /b 0

:download_datasets
call :log "Downloading datasets..."

:: Create data directory
if not exist "data" mkdir data

:: Download datasets
python scripts\prepare_data.py --dataset all
if errorlevel 1 (
    call :error "Failed to download datasets"
    exit /b 1
)
call :success "Downloaded datasets"
exit /b 0

:: Utility functions for colored output
:log
echo %BLUE%%BOLD%=^>%NC% %~1
exit /b 0

:success
echo %GREEN%✓%NC% %~1
exit /b 0

:error
echo %RED%✗ ERROR:%NC% %~1
exit /b 0

:warn
echo %YELLOW%!%NC% %~1
exit /b 0 