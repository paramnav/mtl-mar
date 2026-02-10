# MTL-MAR: Multi-Task Learning for Marine Sedimentation Rate and TOC Prediction

This project implements machine learning models for predicting marine sedimentation rates (MAR) and Total Organic Carbon (TOC) using multi-task learning and various tree-based approaches.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data from Zenodo

**IMPORTANT**: Before running any notebooks, you must download the raw features and labels from Zenodo.

```bash
python setup_project.py
```

Or download directly:

```bash
python download_data_from_zenodo.py
```

**Note**: You need to configure the Zenodo record ID or file URLs in `download_data_from_zenodo.py` after uploading your data. See `README_DATA_DOWNLOAD.md` for detailed instructions.

### 3. Run Notebooks

After data is downloaded, you can run the notebooks:

- `notebooks/train_neural_network.ipynb` - Train neural network models
- `notebooks/train_tree_based_models.ipynb` - Train Random Forest, XGBoost, CatBoost, KNN
- `notebooks/train_multi_task_learning.ipynb` - Train MTL models
- `notebooks/create_training_features_CS.ipynb` - Create training features for CS
- `notebooks/Make_Global_Features.ipynb` - Create global feature maps

## Project Structure

```
mtl-mar/
├── data/
│   ├── raw/              # Raw features and labels (downloaded from Zenodo)
│   ├── interim/           # Processed features and intermediate data
│   └── output/           # Model predictions and outputs
├── notebooks/            # Jupyter notebooks for training and analysis
├── src/
│   ├── models/          # Model training and prediction code
│   ├── preprocessing/   # Data preprocessing utilities
│   └── postprocessing/  # Model evaluation and visualization
├── download_data_from_zenodo.py  # Download script (run first!)
├── setup_project.py      # Complete setup script
└── requirements.txt      # Python dependencies
```

## Data Requirements

The following files must be downloaded from Zenodo:

- **Raw Features**: `data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/` (directory)
- **Raw Labels**: 
  - `data/raw/labels/sedrate_deep.csv`
  - `data/raw/labels/sedrate_continentalshelves.csv`
  - `data/raw/island_map.npy`

See `README_DATA_DOWNLOAD.md` for detailed download instructions.

## Model Types

1. **Deep Learning Models**: TabPFN, Feedforward Neural Networks (NN), and Multi-Task Learning (MTL, shared encoder with task-specific heads for MAR and TOC)
2. **Tree-Based Models**: Random Forest, XGBoost, CatBoost
3. **Distance-Based Models**: K-Nearest Neighbors (kNN)

## Reproducibility

All file paths use relative `Path` objects with `.resolve().parents[X]` for portability. The notebooks will work on any machine with the same directory structure.

## Documentation

- `README_DATA_DOWNLOAD.md` - Instructions for downloading data from Zenodo

## License

This project is licensed under the **MIT License**.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
