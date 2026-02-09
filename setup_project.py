#!/usr/bin/env python
"""
Project Setup Script

This script sets up the project by:
1. Downloading raw features and labels from Zenodo
2. Verifying all required files are present
3. Creating necessary directories

Run this script first before using any notebooks or other scripts.

Usage:
    python setup_project.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check if required Python packages are installed."""
    print("Checking dependencies...")
    
    required_packages = {
        'requests': 'requests',
        'tqdm': 'tqdm',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'xarray': 'xarray',
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print()
        print("=" * 80)
        print("⚠ MISSING DEPENDENCIES")
        print("=" * 80)
        print(f"Please install missing packages:")
        print(f"  pip install {' '.join(missing_packages)}")
        print()
        return False
    
    print("  ✓ All dependencies installed")
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    print()
    print("Creating directories...")
    
    directories = [
        PROJECT_ROOT / "data" / "raw" / "features",
        PROJECT_ROOT / "data" / "raw" / "labels",
        PROJECT_ROOT / "data" / "interim" / "inputfeatures",
        PROJECT_ROOT / "data" / "interim" / "worldfeatures",
        PROJECT_ROOT / "data" / "interim" / "selectedFeatureLists",
        PROJECT_ROOT / "data" / "interim" / "masks",
        PROJECT_ROOT / "data" / "output" / "predictionmaps_SedRate" / "hyperparameter_optimization",
        PROJECT_ROOT / "data" / "figs" / "mar",
        PROJECT_ROOT / "data" / "figs" / "toc",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory.relative_to(PROJECT_ROOT)}")
    
    return True


def download_data():
    """Download data from Zenodo."""
    print()
    print("=" * 80)
    print("Step 1: Download Data from Zenodo")
    print("=" * 80)
    print()
    
    try:
        from download_data_from_zenodo import main as download_main
        return download_main()
    except ImportError as e:
        print(f"  ✗ Error importing download script: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error during download: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 80)
    print("MTL-MAR Project Setup")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Check dependencies")
    print("  2. Create necessary directories")
    print("  3. Download raw features and labels from Zenodo")
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and run again.")
        return False
    
    # Step 2: Create directories
    if not create_directories():
        print("\nError creating directories.")
        return False
    
    # Step 3: Download data
    if not download_data():
        print()
        print("=" * 80)
        print("⚠ DATA DOWNLOAD INCOMPLETE")
        print("=" * 80)
        print()
        print("Please:")
        print("  1. Upload your data to Zenodo")
        print("  2. Update download_data_from_zenodo.py with your Zenodo record ID or URLs")
        print("  3. Run this script again")
        print()
        print("See README_DATA_DOWNLOAD.md for detailed instructions.")
        return False
    
    print()
    print("=" * 80)
    print("✓ PROJECT SETUP COMPLETE")
    print("=" * 80)
    print()
    print("You can now run the notebooks and scripts!")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
