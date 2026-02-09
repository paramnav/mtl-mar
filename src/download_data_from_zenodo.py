#!/usr/bin/env python
"""
Download Raw Features and Labels from Zenodo

This script downloads the raw features and labels dataset from Zenodo
before running any other notebooks or scripts. It should be executed first
to ensure all required data files are available.

Zenodo Repository: https://zenodo.org/records/18545251
DOI: 10.5281/zenodo.18545251

Usage:
    python download_data_from_zenodo.py

The script will:
1. Check if files already exist (skip if present)
2. Download features.zip (12.4 GB) to data/raw/features/
3. Download labels_mar.zip (244.1 kB) to data/raw/labels/
4. Download labels_toc.zip (862.4 kB) to data/raw/labels/
5. Extract all archives automatically
6. Show progress for large downloads
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil

# Get project root directory (go up one level from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"
LABELS_DIR = DATA_RAW / "labels"

# Zenodo configuration
# ============================================================================
# Zenodo repository: https://zenodo.org/records/18545251
# DOI: 10.5281/zenodo.18545251
# ============================================================================
# Option 1: Use Zenodo record ID (recommended)
ZENODO_RECORD_ID = "18545251"  # Zenodo record ID
ZENODO_DOI = "10.5281/zenodo.18545251"  # DOI

# Option 2: Use direct file URLs (alternative)
# Update these with your actual Zenodo file URLs after upload
ZENODO_FILES = {
    # Features directory files
    "features": {
        "url": None,  # Update with Zenodo file URL for features archive
        "filename": "features.zip",
        "extract_to": FEATURES_DIR,
        "description": "Raw feature files (FeaturesPhrampusLee_TOCSedRate_updated)"
    },
    # Label files
    "labels": {
        "url": None,  # Update with Zenodo file URL for labels archive
        "filename": "labels.zip",
        "extract_to": LABELS_DIR,
        "description": "Raw label files (sedrate_deep.csv, sedrate_continentalshelves.csv, island_map.npy)"
    }
}

ZENODO_BASE_URL = "https://zenodo.org/api/records"


def download_file(url, destination, description="File"):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download from
        destination (Path): Destination file path
        description (str): Description for progress bar
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if destination.exists():
        print(f"  ✓ {description} already exists: {destination}")
        return True
    
    print(f"  Downloading {description}...")
    print(f"    URL: {url}")
    print(f"    Destination: {destination}")
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"    Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"  ✓ Successfully downloaded: {destination}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading {description}: {e}")
        return False


def extract_archive(archive_path, extract_to, description="Archive"):
    """
    Extract a zip archive to a directory.
    
    Handles nested directory structures in zip files. If the zip contains
    a single top-level directory, extracts its contents directly to extract_to.
    
    Args:
        archive_path (Path): Path to zip file
        extract_to (Path): Directory to extract to
        description (str): Description for logging
    """
    if not archive_path.exists():
        print(f"  ✗ Archive not found: {archive_path}")
        return False
    
    print(f"  Extracting {description}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get total files for progress
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Check if all files are in a single top-level directory
            if total_files > 0:
                top_level_dirs = set()
                for file in file_list:
                    parts = file.split('/')
                    if len(parts) > 1 and parts[0]:
                        top_level_dirs.add(parts[0])
                
                # If single top-level directory, extract contents directly
                if len(top_level_dirs) == 1:
                    top_dir = list(top_level_dirs)[0]
                    print(f"    Found single top-level directory: {top_dir}")
                    print(f"    Extracting {total_files} files...")
                    with tqdm(total=total_files, desc="    Extracting") as pbar:
                        for file in file_list:
                            # Remove top-level directory from path
                            if file.startswith(top_dir + '/'):
                                target_path = file[len(top_dir) + 1:]
                                if target_path:  # Skip the directory entry itself
                                    target_file = extract_to / target_path
                                    target_file.parent.mkdir(parents=True, exist_ok=True)
                                    with zip_ref.open(file) as source, open(target_file, 'wb') as target:
                                        target.write(source.read())
                            pbar.update(1)
                else:
                    # Multiple top-level items, extract as-is
                    print(f"    Extracting {total_files} files...")
                    with tqdm(total=total_files, desc="    Extracting") as pbar:
                        for file in file_list:
                            zip_ref.extract(file, extract_to)
                            pbar.update(1)
        
        print(f"  ✓ Successfully extracted to: {extract_to}")
        return True
        
    except zipfile.BadZipFile:
        print(f"  ✗ Error: {archive_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"  ✗ Error extracting {description}: {e}")
        return False


def get_zenodo_files(record_id):
    """
    Get file URLs from Zenodo record.
    
    Args:
        record_id (str): Zenodo record ID
        
    Returns:
        dict: Dictionary mapping file names to download URLs
    """
    try:
        url = f"{ZENODO_BASE_URL}/{record_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        files = {}
        
        for file_info in data.get('files', []):
            filename = file_info['key']
            download_url = file_info['links']['self']
            files[filename] = download_url
        
        return files
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching Zenodo record: {e}")
        return {}


def download_from_zenodo_record(record_id, file_patterns=None):
    """
    Download files from a Zenodo record.
    
    Args:
        record_id (str): Zenodo record ID
        file_patterns (dict): Dictionary mapping descriptions to filename patterns
                             e.g., {"features": "features*.zip", "labels_mar": "labels_mar*.zip"}
    """
    print(f"Fetching file list from Zenodo record {record_id}...")
    zenodo_files = get_zenodo_files(record_id)
    
    if not zenodo_files:
        print("  ✗ No files found or error accessing Zenodo record")
        return False
    
    print(f"  Found {len(zenodo_files)} files in Zenodo record")
    print(f"  Files: {', '.join(zenodo_files.keys())}")
    print()
    
    success = True
    
    # Download files based on patterns
    if file_patterns:
        for desc, pattern in file_patterns.items():
            matching_files = [f for f in zenodo_files.keys() if pattern in f.lower()]
            if matching_files:
                # Download the first matching file
                filename = matching_files[0]
                url = zenodo_files[filename]
                # Determine destination based on description
                if desc == "features":
                    dest = DATA_RAW / "features" / filename
                    extract_to = FEATURES_DIR
                elif desc in ["labels_mar", "labels_toc"]:
                    dest = DATA_RAW / "labels" / filename
                    extract_to = LABELS_DIR
                else:
                    dest = DATA_RAW / filename
                    extract_to = None
                
                if download_file(url, dest, f"{desc} ({filename})"):
                    if extract_to and filename.endswith('.zip'):
                        extract_archive(dest, extract_to, desc)
                        # Clean up zip file after extraction
                        dest.unlink()
                else:
                    success = False
            else:
                print(f"  ⚠ No file matching pattern '{pattern}' found for {desc}")
                success = False
    
    return success


def download_from_direct_urls():
    """
    Download files from direct URLs (if configured).
    """
    print("Downloading from direct URLs...")
    print()
    
    success = True
    
    # Download features
    if ZENODO_FILES["features"]["url"]:
        features_info = ZENODO_FILES["features"]
        archive_path = DATA_RAW / "features" / features_info["filename"]
        
        if download_file(features_info["url"], archive_path, features_info["description"]):
            if archive_path.suffix == '.zip':
                extract_archive(archive_path, features_info["extract_to"], features_info["description"])
                # Clean up zip file after extraction
                archive_path.unlink()
        else:
            success = False
    else:
        print("  ⚠ Features URL not configured. Skipping features download.")
    
    # Download labels
    if ZENODO_FILES["labels"]["url"]:
        labels_info = ZENODO_FILES["labels"]
        archive_path = DATA_RAW / "labels" / labels_info["filename"]
        
        if download_file(labels_info["url"], archive_path, labels_info["description"]):
            if archive_path.suffix == '.zip':
                extract_archive(archive_path, labels_info["extract_to"], labels_info["description"])
                # Clean up zip file after extraction
                archive_path.unlink()
        else:
            success = False
    else:
        print("  ⚠ Labels URL not configured. Skipping labels download.")
    
    return success


def verify_downloads():
    """
    Verify that required files have been downloaded.
    
    Returns:
        bool: True if all required files exist
    """
    print()
    print("=" * 80)
    print("Verifying downloaded files...")
    print("=" * 80)
    
    required_files = [
        (FEATURES_DIR, "Features directory", True),  # Directory
        (LABELS_DIR / "sedrate_deep.csv", "Deep ocean MAR labels", False),
        (LABELS_DIR / "sedrate_continentalshelves.csv", "Continental shelf MAR labels", False),
        (LABELS_DIR / "toc_deep_v2.csv", "Deep ocean TOC labels", False),
        (LABELS_DIR / "toc_continentalshelves_v2.csv", "Continental shelf TOC labels", False),
        (DATA_RAW / "island_map.npy", "Island/land mask", False),
    ]
    
    all_present = True
    for file_path, description, is_directory in required_files:
        if file_path.exists():
            if is_directory:
                # Check if directory has files
                file_count = len(list(file_path.glob('*'))) if file_path.is_dir() else 0
                print(f"  ✓ {description}: {file_path} ({file_count} items)")
            else:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {description}: MISSING - {file_path}")
            all_present = False
    
    if all_present:
        print()
        print("  ✓ All required files are present!")
    else:
        print()
        print("  ⚠ Some required files are missing.")
        print("     Please check the download and extraction process.")
    
    return all_present


def main():
    """
    Main function to download data from Zenodo.
    """
    print("=" * 80)
    print("Download Raw Features and Labels from Zenodo")
    print("=" * 80)
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_RAW}")
    print()
    
    # Check if Zenodo record ID is configured
    if ZENODO_RECORD_ID:
        print(f"Using Zenodo record: {ZENODO_RECORD_ID}")
        if ZENODO_DOI:
            print(f"DOI: {ZENODO_DOI}")
        print(f"Repository: https://zenodo.org/records/{ZENODO_RECORD_ID}")
        print()
        success = download_from_zenodo_record(
            ZENODO_RECORD_ID,
            file_patterns={
                "features": "features.zip",
                "labels_mar": "labels_mar.zip",
                "labels_toc": "labels_toc.zip"
            }
        )
    elif any(ZENODO_FILES[k]["url"] for k in ZENODO_FILES):
        print("Using direct file URLs")
        print()
        success = download_from_direct_urls()
    else:
        print("=" * 80)
        print("⚠ CONFIGURATION REQUIRED")
        print("=" * 80)
        print()
        print("Please configure the download script with your Zenodo information:")
        print()
        print("The script is currently configured to use:")
        print(f"  Zenodo Record ID: {ZENODO_RECORD_ID or 'Not set'}")
        print(f"  DOI: {ZENODO_DOI or 'Not set'}")
        print()
        print("If you need to change the repository, update:")
        print("  1. ZENODO_RECORD_ID with your Zenodo record ID")
        print("  2. ZENODO_DOI with your DOI")
        print()
        return False
    
    # Verify downloads
    if success:
        verify_downloads()
        print()
        print("=" * 80)
        print("Download complete!")
        print("=" * 80)
        print()
        print("You can now run the notebooks and scripts.")
    else:
        print()
        print("=" * 80)
        print("⚠ Some downloads failed. Please check the errors above.")
        print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
