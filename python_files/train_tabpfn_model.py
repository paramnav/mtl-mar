"""
Train TabPFN model for sediment rate prediction.

This script trains a TabPFN (Tabular Prior-data Fitted Networks) model
on combined Continental Shelf and Deep Ocean sediment rate data,
evaluates performance, and generates global prediction maps.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from os import walk
import gc

# Third party imports
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path and import utils
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import PROJECT_ROOT, DATA_INTERIM, DATA_RAW, DATA_OUTPUT

# Add models directory to path
module_path = project_root / "src" / "models"
sys.path.append(str(module_path))

from train_neural_network import (
    preprocess_features_labels, split_train_test_data, set_seed
)

# Set random seed
set_seed(42)

# Load world features to get feature dimension
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = sorted([f for f in dataset_path.iterdir() if f.is_file()])
features = np.load(files[0])
world_feature_shape_dim = features.shape[1]

# Load Deep Ocean features
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"
features = np.load(Path(feature_path / "numpy_features.npy"))
labels = np.load(Path(feature_path / "numpy_labels.npy"))
X_mean = np.load(feature_path / "features_mean.npy").astype(float)
X_std = np.load(feature_path / "features_std.npy").astype(float)
lats = np.load(feature_path / "numpy_lat.npy")
lons = np.load(feature_path / "numpy_lon.npy")
input_feature_shape_dim = features.shape[1]

if world_feature_shape_dim != input_feature_shape_dim:
    raise ValueError(
        f"Feature dimensions do not match: World features have {world_feature_shape_dim} "
        f"dimensions while input features have {input_feature_shape_dim} dimensions"
    )

print(f"Number of features: {features.shape[1]}")

features_DO, labels_DO, lats_DO, lons_DO = preprocess_features_labels(
    features, labels, lats, lons, X_mean, X_std
)

# Split DO data
test_size_ratio = 0.25
n_bins = 10
y_binned = pd.qcut(labels_DO, q=n_bins, labels=False, duplicates='drop')
train_features_DO, train_labels_DO, test_features_DO, test_labels_DO = split_train_test_data(
    features_DO, labels_DO, test_size_ratio=test_size_ratio, random_seed=42, stratify=y_binned
)

# Load Continental Shelf features
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_CS_men"
features = np.load(Path(feature_path / "numpy_features.npy"))
labels = np.load(Path(feature_path / "numpy_labels.npy"))
X_mean = np.load(feature_path / "features_mean.npy").astype(float)
X_std = np.load(feature_path / "features_std.npy").astype(float)
lats = np.load(feature_path / "numpy_lat.npy")
lons = np.load(feature_path / "numpy_lon.npy")

features_CS, labels_CS, lats_CS, lons_CS = preprocess_features_labels(
    features, labels, lats, lons, X_mean, X_std
)

# Split CS data
y_binned = pd.qcut(labels_CS, q=n_bins, labels=False, duplicates='drop')
train_features_CS, train_labels_CS, test_features_CS, test_labels_CS = split_train_test_data(
    features_CS, labels_CS, test_size_ratio=test_size_ratio, random_seed=42, stratify=y_binned
)

# Combine CS and DO data
train_features = np.concatenate([train_features_CS, train_features_DO], axis=0)
test_features = np.concatenate([test_features_CS, test_features_DO], axis=0)
train_labels = np.concatenate([train_labels_CS, train_labels_DO], axis=0)
test_labels = np.concatenate([test_labels_CS, test_labels_DO], axis=0)

print(f"Training samples: {len(train_labels)}")
print(f"Test samples: {len(test_labels)}")

# Initialize and train TabPFN regressor
print("Training TabPFN model...")
regressor = TabPFNRegressor(ignore_pretraining_limits=True)
regressor.fit(train_features, train_labels)

# Evaluate on test set
predictions = regressor.predict(test_features)
mse = mean_squared_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# Calculate additional metrics
y_train_pred = regressor.predict(train_features)
train_mae = mean_absolute_error(train_labels, y_train_pred)
train_corr = np.corrcoef(train_labels, y_train_pred)[0, 1]
test_mae = mean_absolute_error(test_labels, predictions)
test_corr = np.corrcoef(test_labels, predictions)[0, 1]

print(f"Train MAE: {train_mae:.4f}")
print(f"Train Correlation: {train_corr:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Correlation: {test_corr:.4f}")

# Generate global prediction map
print("Generating global prediction map...")
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_CS_men"
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []
for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()
chunk_shape = [6, 4320]  # 360 chunks

first_run = True
for count, file in enumerate(files):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)
    
    # Get predictions with uncertainty estimates
    result = regressor.predict(features, output_type="full", quantiles=[0.05, 0.5, 0.95])
    predictions_mean = result["mean"].reshape(chunk_shape)
    quantiles_dict = result["quantiles"]
    predictions_lower = quantiles_dict[0].reshape(chunk_shape)
    predictions_median = quantiles_dict[1].reshape(chunk_shape)
    predictions_upper = quantiles_dict[2].reshape(chunk_shape)
    predictions_var = ((predictions_upper - predictions_lower) / 2) ** 2
    
    if first_run:
        prediction_map = predictions_mean
        prediction_map_var = predictions_var
        first_run = False
    else:
        prediction_map = np.append(prediction_map, predictions_mean, axis=0)
        prediction_map_var = np.append(prediction_map_var, predictions_var, axis=0)
    
    if (count + 1) % 10 == 0:
        print(f"Prediction {count + 1}/{len(files)} done!")

# Rotate and flip prediction map
prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))

# Save prediction map
output_path = (
    DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / 
    "prediction_map_SedRate_entire_tabpfn.npy"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, prediction_map)
print(f"Prediction map saved to: {output_path}")
