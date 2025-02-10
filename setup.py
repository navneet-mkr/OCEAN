#!/usr/bin/env python3
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional
import shutil
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich import print as rprint

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("ocean-setup")
console = Console()

def run_command(cmd: List[str], desc: str) -> bool:
    """Run a command with rich progress indicator"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(desc, total=None)
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            progress.update(task, completed=True)
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                return False
            return True
        except Exception as e:
            progress.update(task, completed=True)
            logger.error(f"Failed to run command: {e}")
            return False

def check_prerequisites() -> bool:
    """Check if required tools are installed"""
    prerequisites = {
        "python": "Python 3.8+",
        "git": "Git",
        "git-lfs": "Git LFS"
    }
    
    missing = []
    for cmd, name in prerequisites.items():
        if not shutil.which(cmd):
            missing.append(name)
    
    if missing:
        logger.error("[red]Missing prerequisites:[/red]")
        for item in missing:
            logger.error(f"  - {item}")
        return False
    return True

def setup_environment() -> bool:
    """Setup virtual environment and install dependencies"""
    logger.info("[bold blue]Setting up Python environment...[/bold blue]")
    
    # Create virtual environment
    if not run_command(
        [sys.executable, "-m", "venv", ".venv"],
        "Creating virtual environment..."
    ):
        return False
    
    # Determine activation script
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
    else:
        activate_script = "source .venv/bin/activate"
    
    logger.info(f"[green]Virtual environment created. Activate it with: {activate_script}[/green]")
    
    # Install dependencies
    if not run_command(
        [".venv/bin/pip" if sys.platform != "win32" else ".venv\\Scripts\\pip",
         "install", "-r", "requirements.txt"],
        "Installing dependencies..."
    ):
        return False
    
    logger.info("[green]Dependencies installed successfully![/green]")
    return True

def setup_git_lfs() -> bool:
    """Setup Git LFS"""
    logger.info("[bold blue]Setting up Git LFS...[/bold blue]")
    return run_command(["git", "lfs", "install"], "Installing Git LFS...")

def setup_huggingface() -> bool:
    """Setup Hugging Face authentication"""
    logger.info("[bold blue]Setting up Hugging Face authentication...[/bold blue]")
    logger.info("Please follow the instructions to log in to Hugging Face:")
    return run_command(["huggingface-cli", "login"], "Logging in to Hugging Face...")

def download_datasets() -> bool:
    """Download and prepare datasets"""
    logger.info("[bold blue]Downloading datasets...[/bold blue]")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Run prepare_data.py script
    return run_command(
        [sys.executable, "scripts/prepare_data.py", "--dataset", "all"],
        "Downloading and preparing datasets..."
    )

def main():
    console.rule("[bold blue]OCEAN Setup[/bold blue]")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("[red]Please install missing prerequisites and try again.[/red]")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Setting up Python environment", setup_environment),
        ("Setting up Git LFS", setup_git_lfs),
        ("Setting up Hugging Face", setup_huggingface),
        ("Downloading datasets", download_datasets)
    ]
    
    # Run setup steps
    for desc, func in steps:
        console.rule(f"[bold cyan]{desc}[/bold cyan]")
        if not func():
            logger.error(f"[red]Failed to complete: {desc}[/red]")
            sys.exit(1)
        logger.info(f"[green]✓ {desc} completed successfully![/green]")
    
    console.rule("[bold green]Setup Complete![/bold green]")
    logger.info("""
[bold green]OCEAN is ready to use![/bold green]

To get started:
1. Activate the virtual environment:
   [yellow]source .venv/bin/activate[/yellow]  # On Unix
   [yellow].venv\\Scripts\\activate[/yellow]    # On Windows

2. Run experiments:
   [yellow]python scripts/run_experiments.py[/yellow]

For more information, please refer to the README.md file.
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\n[red]Setup interrupted by user.[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[red]Setup failed: {str(e)}[/red]")
        sys.exit(1) 