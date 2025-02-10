#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print with color and formatting
log() {
    echo -e "${BLUE}${BOLD}=>${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}!${NC} $1"
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python is not installed"
    fi
    if ! python3 -c "import sys; assert sys.version_info >= (3,8)" &> /dev/null; then
        error "Python 3.8+ is required"
    fi
    success "Python 3.8+ is installed"

    # Check Git
    if ! command -v git &> /dev/null; then
        error "Git is not installed"
    fi
    success "Git is installed"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip is not installed"
    fi
    success "pip is installed"
}

install_git_lfs() {
    log "Installing Git LFS..."

    # Check if Git LFS is already installed
    if command -v git-lfs &> /dev/null; then
        success "Git LFS is already installed"
        return
    fi

    # Try to install Git LFS based on the OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install git-lfs
        else
            warn "Please install Git LFS manually using Homebrew:"
            warn "brew install git-lfs"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v yum &> /dev/null; then
            sudo yum install -y git-lfs
        else
            warn "Please install Git LFS manually using your package manager"
            exit 1
        fi
    else
        warn "Please install Git LFS manually from: https://git-lfs.com"
        exit 1
    fi

    git lfs install
    success "Git LFS installed and configured"
}

setup_python_env() {
    log "Setting up Python environment..."

    # Create virtual environment
    python3 -m venv .venv || error "Failed to create virtual environment"
    success "Created virtual environment"

    # Activate virtual environment
    source .venv/bin/activate || error "Failed to activate virtual environment"
    success "Activated virtual environment"

    # Upgrade pip
    python3 -m pip install --upgrade pip || error "Failed to upgrade pip"

    # Install dependencies
    pip install -r requirements.txt || error "Failed to install dependencies"
    success "Installed dependencies"
}

setup_huggingface() {
    log "Setting up Hugging Face..."

    # Check if huggingface-cli is installed
    if ! command -v huggingface-cli &> /dev/null; then
        warn "huggingface-cli not found. Installing..."
        pip install --upgrade "huggingface_hub[cli]"
    fi

    warn "Please follow the instructions to log in to Hugging Face:"
    huggingface-cli login || error "Failed to login to Hugging Face"
    success "Hugging Face setup complete"
}

download_datasets() {
    log "Downloading datasets..."

    # Create data directory
    mkdir -p data

    # Download datasets
    python3 scripts/prepare_data.py --dataset all || error "Failed to download datasets"
    success "Downloaded datasets"
}

main() {
    log "Setting up OCEAN..."

    check_prerequisites
    install_git_lfs
    setup_python_env
    setup_huggingface
    download_datasets

    echo
    success "Setup Complete!"
    echo
    echo "To get started:"
    echo "1. Activate the virtual environment:"
    echo "   ${YELLOW}source .venv/bin/activate${NC}"
    echo
    echo "2. Run experiments:"
    echo "   ${YELLOW}python scripts/run_experiments.py${NC}"
    echo
    echo "For more information, please refer to the README.md file."
}

main 