# Data Download Setup

## Overview

This project requires raw features and labels to be downloaded from Zenodo before running any notebooks or scripts. The download script (`download_data_from_zenodo.py`) handles this automatically.

## Prerequisites

Install required packages:

```bash
pip install requests tqdm
```

## Setup Instructions

### Step 1: Upload Data to Zenodo

1. Create a Zenodo account at https://zenodo.org
2. Create a new upload/deposit
3. Upload your data files:
   - **Features archive**: All files from `data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/` (upload as `features.zip`)
   - **Labels archive**: Files from `data/raw/labels/` including:
     - `sedrate_deep.csv`
     - `sedrate_continentalshelves.csv`
     - `island_map.npy`
4. Publish the deposit to get a DOI and record ID

### Step 2: Configure Download Script

Edit `download_data_from_zenodo.py` and update one of these options:

**Option A: Using Zenodo Record ID (Recommended)**
```python
ZENODO_RECORD_ID = "1234567"  # Replace with your record ID
```

**Option B: Using Direct File URLs**
```python
ZENODO_FILES = {
    "features": {
        "url": "https://zenodo.org/record/18545251/files/features.zip",
        ...
    },
    "labels": {
        "url": "https://zenodo.org/record/18545251/files/labels.zip",
        ...
    }
}
```

### Step 3: Run Download Script

```bash
cd /path/to/mtl-mar
python download_data_from_zenodo.py
```

The script will:
- Check if files already exist (skips if present)
- Download features and labels archives
- Extract files to correct directories
- Verify all required files are present

## Required Files

After download, the following files should exist:

- `data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/` (directory with all feature files)
- `data/raw/labels/sedrate_deep.csv`
- `data/raw/labels/sedrate_continentalshelves.csv`
- `data/raw/island_map.npy`

## Troubleshooting

- **Connection errors**: Check your internet connection and Zenodo availability
- **File not found**: Verify the Zenodo record ID or URLs are correct
- **Extraction errors**: Ensure the downloaded files are valid zip archives
- **Missing files**: Check that all files were uploaded to Zenodo correctly
