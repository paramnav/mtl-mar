#!/usr/bin/env python
# coding: utf-8
# Converted from Jupyter notebook: Sed_RF_KNN_entire.ipynb
# This script contains all code cells from the notebook


# ======================================================================
# Cell 1
# ======================================================================
# Standard library imports
from pathlib import Path
import sys
import random

# Third party imports
import numpy as np
# import matplotlib.pyplot as plt
import xgboost as xgb
import catboost as cb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Add project root to path and import utils
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import PROJECT_ROOT, DATA_INTERIM, DATA_RAW, DATA_OUTPUT

# Add models directory to path
module_path = project_root / "src" / "models"
sys.path.append(str(module_path))

from trainDNN import *
from explainDNN import *
from predictDNN import *
from treeBasedModel import *

# Add postprocessing directory to path
module_path = project_root / "src" / "postprocessing"
sys.path.append(str(module_path))

from getModelPerformance import *


# ======================================================================
# Cell 2
# ======================================================================
set_seed()


# ======================================================================
# Cell 3
# ======================================================================
# Load world features
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = sorted([f for f in dataset_path.iterdir() if f.is_file()])
features = np.load(files[0])
world_feature_shape_dim = features.shape[1]


#  Load input feature 
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"
features = np.load(Path(feature_path / "numpy_features.npy"))
labels = np.load(Path(feature_path / "numpy_labels.npy"))
X_mean = np.load(feature_path / "features_mean.npy").astype(float)
X_std = np.load(feature_path / "features_std.npy").astype(float)
lats = np.load(feature_path / "numpy_lat.npy")
lons = np.load(feature_path / "numpy_lon.npy")
input_feature_shape_dim = features.shape[1]


if world_feature_shape_dim != input_feature_shape_dim:
    raise ValueError(f"Feature dimensions do not match: World features have {world_feature_shape_dim} dimensions while input features have {input_feature_shape_dim} dimensions")

print(f"Number of features: {features.shape[1]}")


features_DO, labels_DO, lats_DO, lons_DO = preprocess_features_labels(features, labels, lats, lons, X_mean, X_std)

print(X_mean)
print(X_std)

from sklearn.model_selection import StratifiedKFold


test_size_ratio = 0.25
print("Test size ratio") 
print(test_size_ratio)

# Create bins for stratification
n_bins = 10
y_binned = pd.qcut(labels_DO, q=n_bins, labels=False, duplicates='drop')
train_features_DO, train_labels_DO, test_features_DO, test_labels_DO = split_train_test_data(features_DO, labels_DO, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
train_lats_DO, train_lons_DO, test_lats_DO, test_lons_DO = split_train_test_data(lats_DO, lons_DO, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)


# ======================================================================
# Cell 4
# ======================================================================
# Load world features
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = sorted([f for f in dataset_path.iterdir() if f.is_file()])
features = np.load(files[0])
world_feature_shape_dim = features.shape[1]


#  Load input feature 
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_CS_men"
features = np.load(Path(feature_path / "numpy_features.npy"))
labels = np.load(Path(feature_path / "numpy_labels.npy"))
X_mean = np.load(feature_path / "features_mean.npy").astype(float)
X_std = np.load(feature_path / "features_std.npy").astype(float)
lats = np.load(feature_path / "numpy_lat.npy")
lons = np.load(feature_path / "numpy_lon.npy")
input_feature_shape_dim = features.shape[1]


if world_feature_shape_dim != input_feature_shape_dim:
    raise ValueError(f"Feature dimensions do not match: World features have {world_feature_shape_dim} dimensions while input features have {input_feature_shape_dim} dimensions")

print(f"Number of features: {features.shape[1]}")

print(X_mean)
print(X_std)

features_CS, labels_CS, lats_CS, lons_CS = preprocess_features_labels(features, labels, lats, lons, X_mean, X_std)


from sklearn.model_selection import StratifiedKFold


test_size_ratio = 0.25
print("Test size ratio") 
print(test_size_ratio)

# Create bins for stratification
n_bins = 10
y_binned = pd.qcut(labels_CS, q=n_bins, labels=False, duplicates='drop')
train_features_CS, train_labels_CS, test_features_CS, test_labels_CS = split_train_test_data(features_CS, labels_CS, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
train_lats_CS, train_lons_CS, test_lats_CS, test_lons_CS = split_train_test_data(lats_CS, lons_CS, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)


# ======================================================================
# Cell 5
# ======================================================================
train_features = np.concatenate([train_features_CS, train_features_DO], axis = 0)
test_features = np.concatenate([test_features_CS, test_features_DO], axis = 0)
train_labels = np.concatenate([train_labels_CS, train_labels_DO], axis = 0)
test_labels = np.concatenate([test_labels_CS, test_labels_DO], axis = 0)
features = np.concatenate([features_CS, features_DO], axis = 0)
labels = np.concatenate([labels_CS, labels_DO], axis = 0)


# ======================================================================
# Cell 6
# ======================================================================
# import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
# import matplotlib.ticker as mtick
# from matplotlib.patches import Patch

# fig, ax = plt.subplots(figsize=(3, 1.5), dpi=300)

# KDE for train and test labels
kde_train = gaussian_kde(train_labels)
kde_test = gaussian_kde(test_labels)
x_min = min(np.min(train_labels), np.min(test_labels))
x_max = max(np.max(train_labels), np.max(test_labels))
x_range = np.linspace(x_min, x_max, 300)

# Plot KDEs
# l1, = ax.plot(x_range, kde_train(x_range), label=f'Train KDE (n={len(train_labels)})', color='navy', lw=1.5, alpha=1)
# l2, = ax.plot(x_range, kde_test(x_range), label=f'Test KDE (n={len(test_labels)})', color='darkorange', lw=0.8, alpha=1)

# Make histogram bars thinner using 'histtype="stepfilled"' and 'rwidth' < 1
# hist_train = ax.hist(
#     train_labels, 
#     bins=100, 
#     density=True, 
#     alpha=0.35, 
#     color='navy', 
#     edgecolor='none', 
#     label='Train Histogram',
#     histtype='stepfilled',
#     rwidth=0.2   # 0.2 makes the bars very thin
# )
# hist_test = ax.hist(
#     test_labels, 
#     bins=100, 
#     density=True, 
#     alpha=0.35, 
#     color='darkorange', 
#     edgecolor='none', 
#     label='Test Histogram',
#     histtype='stepfilled',
#     rwidth=0.2
# )

# Legend: add patches for histogram shading
hist_patches = [
    Patch(facecolor='navy', alpha=0.35, label='Train Histogram'),
    Patch(facecolor='darkorange', alpha=0.35, label='Test Histogram')
]

# ax.set_xlim(left=-60, right=1600)
# ax.set_ylim(0, 0.01)

# ax.set_xlabel('MAR data', fontsize=8)
# ax.set_ylabel('Density', fontsize=8)
# ax.tick_params(axis='both', which='major', labelsize=7)
# ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
# ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

# Build joint legend for KDE lines and histogram shadings
legend_handles = [l1, l2] + hist_patches
# ax.legend(handles=legend_handles, frameon=False, fontsize=7, loc='upper right')

# ax.grid(True, which='major', ls=':', lw=0.3, alpha=0.5)

# plt.tight_layout(pad=0.2)
# plt.show()


# ======================================================================
# Cell 7
# ======================================================================
import geopandas as gpd
# import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Prepare training and testing coordinates and labels
train_lats = np.concatenate([train_lats_CS, train_lats_DO])
train_lons = np.concatenate([train_lons_CS, train_lons_DO])
test_lats = np.concatenate([test_lats_CS, test_lats_DO])
test_lons = np.concatenate([test_lons_CS, test_lons_DO])

train_labels_plot = np.concatenate([train_labels_CS, train_labels_DO])
test_labels_plot = np.concatenate([test_labels_CS, test_labels_DO])

# Create GeoDataFrames for train and test
train_gdf = gpd.GeoDataFrame({
    'label': train_labels_plot,
    'dataset': 'train'
}, geometry=[Point(lon, lat) for lon, lat in zip(train_lons, train_lats)], crs="EPSG:4326")

test_gdf = gpd.GeoDataFrame({
    'label': test_labels_plot,
    'dataset': 'test'
}, geometry=[Point(lon, lat) for lon, lat in zip(test_lons, test_lats)], crs="EPSG:4326")

# Combine for plotting
all_gdf = pd.concat([train_gdf, test_gdf], ignore_index=True)

# --- Overlay bathymetry regions: deep ocean (<-200m, blue) and shelf (-200m to 0m, brown) ---

restrepo_features_path = DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"
etopo_file = Path(restrepo_features_path) / "GL_ELEVATION_M_ASL_ETOPO2v2.5m.grd"

# Try to load the bathymetry data and plot the regions
bathymetry_plotted = False
try:
    etopo_ds = xr.open_dataset(etopo_file, format='NETCDF4')
    # Find the elevation variable
    elev_var = [v for v in etopo_ds.data_vars if 'elev' in v.lower() or v.lower() in ['z', 'elevation']]
    if elev_var:
        elev = etopo_ds[elev_var[0]]
    else:
        elev = list(etopo_ds.data_vars.values())[0]  # fallback to first variable

    # Get lat/lon grid
    lat_vals = etopo_ds['lat'].values
    lon_vals = etopo_ds['lon'].values

    # Make meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    elev_grid = elev.values

    # Mask for deep ocean (<-200m) and shelf (-200m to 0m)
    deep_ocean_mask = elev_grid < -200
    shelf_mask = (elev_grid >= -200) & (elev_grid < 0)

    # --- FIX: Use correct colormap for shelf region ---
    # The issue is that pcolormesh with color='saddlebrown' and a boolean mask
    # will color all "True" as brown, but "False" as brown as well if not using a colormap.
    # Instead, use cmap with a ListedColormap for shelf, and mask out non-shelf.

    # from matplotlib.colors import ListedColormap

    # fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    # Plot world basemap
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world.plot(ax=ax, color='lightgray', edgecolor='lightgray')

    # Plot deep ocean region (blue, transparent)
    # ax.pcolormesh(
    #     lon_grid, lat_grid, np.where(deep_ocean_mask, 1, np.nan),
    #     cmap=ListedColormap(['#1f77b4']),  # blue
    #     alpha=0.1, shading='auto'
    # )

    # Plot shelf region (brown, transparent)
    # ax.pcolormesh(
        # lon_grid, lat_grid, np.where(shelf_mask, 1, np.nan),
        cmap=ListedColormap(['saddlebrown']),
        # alpha=0.1, shading='auto'
    )

    # Plot training and testing points
    # train_gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label='Train', alpha=0.7)
    # test_gdf.plot(ax=ax, marker='o', color='red', markersize=5, label='Test', alpha=0.7)

    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    bathymetry_plotted = True
except Exception as e:
    print(f"Could not plot bathymetry regions: {e}")

if not bathymetry_plotted:
    # Fallback: just plot the points and world map
    # fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world.plot(ax=ax, color='lightgray', edgecolor='lightgray')
    # train_gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label='Train', alpha=0.7)
    # test_gdf.plot(ax=ax, marker='o', color='red', markersize=5, label='Test', alpha=0.7)
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


# ======================================================================
# Cell 8
# ======================================================================
import geopandas as gpd
# import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Prepare training and testing coordinates and labels
train_lats = np.concatenate([train_lats_CS, train_lats_DO])
train_lons = np.concatenate([train_lons_CS, train_lons_DO])
test_lats = np.concatenate([test_lats_CS, test_lats_DO])
test_lons = np.concatenate([test_lons_CS, test_lons_DO])

train_labels_plot = np.concatenate([train_labels_CS, train_labels_DO])
test_labels_plot = np.concatenate([test_labels_CS, test_labels_DO])

# Create GeoDataFrames for train and test
train_gdf = gpd.GeoDataFrame({
    'label': train_labels_plot,
    'dataset': 'train'
}, geometry=[Point(lon, lat) for lon, lat in zip(train_lons, train_lats)], crs="EPSG:4326")

test_gdf = gpd.GeoDataFrame({
    'label': test_labels_plot,
    'dataset': 'test'
}, geometry=[Point(lon, lat) for lon, lat in zip(test_lons, test_lats)], crs="EPSG:4326")

# Combine for plotting
all_gdf = pd.concat([train_gdf, test_gdf], ignore_index=True)

# --- Overlay bathymetry regions: deep ocean (<-200m, blue) and shelf (-200m to 0m, brown) ---

restrepo_features_path = DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"
etopo_file = Path(restrepo_features_path) / "GL_ELEVATION_M_ASL_ETOPO2v2.5m.grd"

# Try to load the bathymetry data and plot the regions
bathymetry_plotted = False
try:
    etopo_ds = xr.open_dataset(etopo_file, format='NETCDF4')
    # Find the elevation variable
    elev_var = [v for v in etopo_ds.data_vars if 'elev' in v.lower() or v.lower() in ['z', 'elevation']]
    if elev_var:
        elev = etopo_ds[elev_var[0]]
    else:
        elev = list(etopo_ds.data_vars.values())[0]  # fallback to first variable

    # Get lat/lon grid
    lat_vals = etopo_ds['lat'].values
    lon_vals = etopo_ds['lon'].values

    # Make meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    elev_grid = elev.values

    # Mask for deep ocean (<-200m) and shelf (-200m to 0m)
    deep_ocean_mask = elev_grid < -200
    shelf_mask = (elev_grid >= -200) & (elev_grid < 0)

    # --- FIX: Use correct colormap for shelf region ---
    # The issue is that pcolormesh with color='saddlebrown' and a boolean mask
    # will color all "True" as brown, but "False" as brown as well if not using a colormap.
    # Instead, use cmap with a ListedColormap for shelf, and mask out non-shelf.

    # from matplotlib.colors import ListedColormap

    # fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    # Plot world basemap
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world.plot(ax=ax, color='lightgray', edgecolor='lightgray')

    # Plot deep ocean region (blue, transparent)
    # ax.pcolormesh(
    #     lon_grid, lat_grid, np.where(deep_ocean_mask, 1, np.nan),
    #     cmap=ListedColormap(['#1f77b4']),  # blue
    #     alpha=0.1, shading='auto'
    # )

    # Plot shelf region (brown, transparent)
    # ax.pcolormesh(
        # lon_grid, lat_grid, np.where(shelf_mask, 1, np.nan),
        cmap=ListedColormap(['saddlebrown']),
        # alpha=0.1, shading='auto'
    )

    # Plot training and testing points
    # train_gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label='Train', alpha=0.7)
    # test_gdf.plot(ax=ax, marker='o', color='red', markersize=5, label='Test', alpha=0.7)

    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    bathymetry_plotted = True
except Exception as e:
    print(f"Could not plot bathymetry regions: {e}")

if not bathymetry_plotted:
    # Fallback: just plot the points and world map
    # fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world.plot(ax=ax, color='lightgray', edgecolor='lightgray')
    # train_gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label='Train', alpha=0.7)
    # test_gdf.plot(ax=ax, marker='o', color='red', markersize=5, label='Test', alpha=0.7)
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.set_xlim(-180, 180)
    # ax.set_ylim(-90, 90)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


# ======================================================================
# Cell 9
# ======================================================================
# plt.imshow(deep_ocean_mask)


# ======================================================================
# Cell 10
# ======================================================================
num_trees = np.concatenate([np.linspace(0,10, 11, dtype = int)[1:], np.linspace(0, 100, 11, dtype = int)[2:], np.linspace(0, 1000, 11, dtype = int)[2:]])


# ======================================================================
# Cell 11
# ======================================================================
n_estimators_list = []
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []
train_corr_list = []
test_corr_list = []

# Loop over n_estimators
for n_estimators in num_trees:
    print("="*80)
    print(f"Random Forest with {n_estimators} trees")
    
    # Train and evaluate
    model, training_mae, evaluation_mae, train_predictions, test_predictions = train_tree_based_model(
        train_features, train_labels, test_features, test_labels,
        n_estimators=n_estimators
    )
    # Collect performance metrics
    n_estimators_list.append(n_estimators)

    train_mae, train_corr, train_r2 = create_scatter_plot(train_labels, train_predictions, "dummy_train.png")
    test_mae, test_corr, test_r2 = create_scatter_plot(test_labels, test_predictions, "dummy_test.png")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train Correlation: {train_corr:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_corr_list.append(train_corr)
    test_corr_list.append(test_corr)


plot_tree_model_performance(
    n_estimators_list,
    train_mae_list, test_mae_list,
    train_r2_list, test_r2_list,
    train_corr_list, test_corr_list
)


# ======================================================================
# Cell 12
# ======================================================================

#set_seed(42)
model_type = "xgboost"
n_estimators_list = []
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []
train_corr_list = []
test_corr_list = []


# Loop over n_estimators
for n_estimators in num_trees:
    set_seed(42)
    print("="*80)
    print(f"XGBoost with {n_estimators} trees")
    model, training_mae, evaluation_mae, train_predictions, test_predictions = train_tree_based_model(
        train_features, train_labels, test_features, test_labels,
        model_type=model_type, 
        n_estimators=n_estimators
    )
    # Collect performance metrics
    n_estimators_list.append(n_estimators)
    train_mae, train_corr, train_r2 = create_scatter_plot(train_labels, train_predictions, "dummy_train.png")
    test_mae, test_corr, test_r2 = create_scatter_plot(test_labels, test_predictions, "dummy_test.png")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train Correlation: {train_corr:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_corr_list.append(train_corr)
    test_corr_list.append(test_corr)


plot_tree_model_performance(
    n_estimators_list,
    train_mae_list, test_mae_list,
    train_r2_list, test_r2_list,
    train_corr_list, test_corr_list
)


# ======================================================================
# Cell 13
# ======================================================================
# To store results
set_seed(42)
model_type = "catboost"
n_estimators_list = []
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []
train_corr_list = []
test_corr_list = []


# Loop over n_estimators
for n_estimators in num_trees:
    set_seed(42)
    print("="*80)
    print(f"CatBoost with {n_estimators} trees")
    model, training_mae, evaluation_mae, train_predictions, test_predictions = train_tree_based_model(
        train_features, train_labels, test_features, test_labels,
        model_type=model_type, 
        n_estimators=n_estimators)
    # Collect performance metrics
    n_estimators_list.append(n_estimators)

    train_mae, train_corr, train_r2 = create_scatter_plot(train_labels, train_predictions, "dummy_train.png")
    test_mae, test_corr, test_r2 = create_scatter_plot(test_labels, test_predictions, "dummy_test.png")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train Correlation: {train_corr:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_corr_list.append(train_corr)
    test_corr_list.append(test_corr)



plot_tree_model_performance(
    n_estimators_list,
    train_mae_list, test_mae_list,
    train_r2_list, test_r2_list,
    train_corr_list, test_corr_list
)


# ======================================================================
# Cell 15
# ======================================================================
# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
from tabpfn import TabPFNRegressor  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load Boston Housing data
from sklearn.datasets import make_regression
import pandas as pd


# Initialize the regressor
regressor = TabPFNRegressor()  
regressor.fit(train_features, train_labels)

# Predict on the test set
predictions = regressor.predict(test_features)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

print("Mean Squared Error (MSE):", mse)


# ======================================================================
# Cell 16
# ======================================================================
from sklearn.metrics import mean_absolute_error
import numpy as np

# Predictions
y_train_pred = regressor.predict(train_features)
y_test_pred = predictions  # Assuming this is regressor.predict(X_test)

# Train metrics
train_mae = mean_absolute_error(train_labels, y_train_pred)
train_corr = np.corrcoef(train_labels, y_train_pred)[0, 1]

# Test metrics
test_mae = mean_absolute_error(test_labels, y_test_pred)
test_corr = np.corrcoef(test_labels, y_test_pred)[0, 1]

# Print metrics
print(f"Train MAE: {train_mae:.4f}")
print(f"Train Correlation: {train_corr:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Correlation: {test_corr:.4f}")


# ======================================================================
# Cell 17
# ======================================================================
# Plotting the results
# plt.figure(figsize=(10, 6))

# Scatter plot for training data
# plt.scatter(train_labels, regressor.predict(train_features), color='blue', label='Train', alpha=0.7)
# Scatter plot for test data
# plt.scatter(test_labels, predictions, color='red', label='Test', alpha=0.7)

# Calculate and display the correlation coefficient
correlation = np.corrcoef(test_labels, predictions)[0, 1]
# plt.text(0.1, 0.9, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12)

# plt.title('TabPFN: Predicted vs. Actual Values')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.legend()
# plt.grid(True)
# plt.show()


# ======================================================================
# Cell 18
# ======================================================================
from os import walk  
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_Sedrate_DO_men"
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

#dataset_path = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "WorldFeaturesAll")#ChlorWorldFeatures
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []

for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()


features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks

np.random.seed(42)


first_run = True
ii = 0
features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks


first_run = True
ii = 0

for count, file in enumerate(files):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)


    # Perform multiple forward passes with dropout
    num_samples = 5  # Adjust this based on your requirements
    predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

    with torch.no_grad():
        for i in range(num_samples):
            predictions = regressor.predict(features) #decoded, 

            # Store predictions for later averaging
            predictions_sum[i] = predictions.reshape(chunk_shape)

    # Average predictions over all samples
    predictions_avg = np.mean(predictions_sum, axis=0)
    predictions_var = np.var(predictions_sum, axis=0)

    if first_run:
        prediction_map2 = predictions_avg
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
        prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)

    ii += 1
    print("Prediction " + str(ii) + " done!")

#del features
gc.collect()

land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1
#prediction_map2[land_map] = 0

#plt.figure(figsize=[25,15], dpi=200)
# import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
# plt.figure(figsize=[25, 15], dpi=50)
prediction_map2[land_map] = 0
# plt.imshow(prediction_map2 , vmin = 0, vmax = 1000, cmap="cubehelix")
# cbar = plt.colorbar()
#"""

#np.save("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_SedRate/hyperparameter_optimization/prediction_map_SedRate_DO_catboost", prediction_map2)


# ======================================================================
# Cell 20
# ======================================================================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
import numpy as np

def train_knn(features_train, labels_train, features_test, labels_test, n_neighbors=5):
    # Initialize the model
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights='uniform',
        metric='minkowski',
        p=2
    )
    
    # Train the model
    model.fit(features_train, labels_train)
    
    # Evaluate on training set
    train_predictions = model.predict(features_train)
    train_mae = mean_absolute_error(labels_train, train_predictions)
    
    # Evaluate on test set
    test_predictions = model.predict(features_test)
    test_mae = mean_absolute_error(labels_test, test_predictions)
    
    return model, train_mae, test_mae, train_predictions, test_predictions


# ======================================================================
# Cell 21
# ======================================================================
# Store results
set_seed(42)
k_values = []
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []
train_corr_list = []
test_corr_list = []

# Values of k to evaluate
k_range = list(range(1, 31)) + list(range(40, 101, 10))  # 1–30, then 40–100 step 10

for k in k_range:
    print("="*80)
    print(f"KNN with k={k} neighbors")

    # Train and evaluate
    model, training_mae, evaluation_mae, train_predictions, test_predictions = train_knn(
        train_features, train_labels, test_features, test_labels,
        n_neighbors=k
    )

    # Compute metrics
    train_r2 = r2_score(train_labels, train_predictions)
    test_r2 = r2_score(test_labels, test_predictions)
    train_corr = np.corrcoef(train_labels.ravel(), train_predictions.ravel())[0, 1]
    test_corr = np.corrcoef(test_labels.ravel(), test_predictions.ravel())[0, 1]

    print(f"Train MAE: {training_mae:.4f}")
    print(f"Train Correlation: {train_corr:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {evaluation_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # Store
    k_values.append(k)
    train_mae_list.append(training_mae)
    test_mae_list.append(evaluation_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_corr_list.append(train_corr)
    test_corr_list.append(test_corr)


plot_tree_model_performance(
    k_values,
    train_mae_list, test_mae_list,
    train_r2_list, test_r2_list,
    train_corr_list, test_corr_list
)


# ======================================================================
# Cell 23
# ======================================================================
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
import numpy as np

def train_lgbm(features_train, labels_train, features_test, labels_test, 
               n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42):
    # Initialize the model
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state
    )
    
    # Train the model
    model.fit(features_train, labels_train)
    
    # Evaluate on training set
    train_predictions = model.predict(features_train)
    train_mae = mean_absolute_error(labels_train, train_predictions)
    
    # Evaluate on test set
    test_predictions = model.predict(features_test)
    test_mae = mean_absolute_error(labels_test, test_predictions)
    
    return model, train_mae, test_mae, train_predictions, test_predictions


# ======================================================================
# Cell 24
# ======================================================================
# Store results for LightGBM
set_seed(42)
n_estimators_values = list(range(10, 201, 10))  # Try n_estimators from 10 to 200 step 10
train_mae_list = []
test_mae_list = []
train_r2_list = []
test_r2_list = []
train_corr_list = []
test_corr_list = []

for n_estimators in n_estimators_values:
    print("="*80)
    print(f"LightGBM with n_estimators={n_estimators}")

    # Train and evaluate
    model, training_mae, evaluation_mae, train_predictions, test_predictions = train_lgbm(
        train_features, train_labels, test_features, test_labels,
        n_estimators=n_estimators
    )

    # Compute metrics
    train_r2 = r2_score(train_labels, train_predictions)
    test_r2 = r2_score(test_labels, test_predictions)
    train_corr = np.corrcoef(train_labels.ravel(), train_predictions.ravel())[0, 1]
    test_corr = np.corrcoef(test_labels.ravel(), test_predictions.ravel())[0, 1]

    print(f"Train MAE: {training_mae:.4f}")
    print(f"Train Correlation: {train_corr:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {evaluation_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # Store
    train_mae_list.append(training_mae)
    test_mae_list.append(evaluation_mae)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    train_corr_list.append(train_corr)
    test_corr_list.append(test_corr)

plot_tree_model_performance(
    n_estimators_values,
    train_mae_list, test_mae_list,
    train_r2_list, test_r2_list,
    train_corr_list, test_corr_list
)


# ======================================================================
# Cell 27
# ======================================================================
model, train_mae, test_mae, train_preds, test_preds = train_tree_based_model(
    train_features, train_labels, test_features, test_labels, n_estimators=60)


# ======================================================================
# Cell 28
# ======================================================================
def plot_predictions(y_train, train_preds, y_test, test_preds, model_name, n_estimators):
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0,1]
    test_corr = np.corrcoef(y_test, test_preds)[0,1]
    
    # Scatter plot
    # plt.figure(figsize=(7,7))
    # plt.scatter(y_train, train_preds, color='blue', label=f'Train (MAE={train_mae:.3f}, CC={train_corr:.3f})', alpha=0.6)
    # plt.scatter(y_test, test_preds, color='orange', label=f'Test (MAE={test_mae:.3f}, CC={test_corr:.3f})', alpha=0.6)
    max_val = max(y_train.max(), y_test.max(), train_preds.max(), test_preds.max())
    min_val = min(y_train.min(), y_test.min(), train_preds.min(), test_preds.min())
    # plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title(f"{model_name} Predictions (n_estimators={n_estimators:.0f})")
    # plt.legend()
    # plt.grid(True)
    
    # Add a text box for metrics
    #textstr = f"Train MAE: {train_mae:.4f}\nTrain corr coeff: {train_corr:.4f}\nTest MAE: {test_mae:.4f}\nTest corr coeff: {test_corr:.4f}"
    #plt.gcf().text(0.65, 0.15, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # plt.show()


# ======================================================================
# Cell 29
# ======================================================================
plot_predictions(train_labels, train_preds, test_labels, test_preds, model_name="random forests", n_estimators = 60)


# ======================================================================
# Cell 30
# ======================================================================
#os.chdir("/gxfs_work1/cau/sunms498/work/david/NN4SedRatePred/models/MLP/SedRate_MAR/models_torch/")

from os import walk  
from tqdm import tqdm  # Progress bar

feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

#dataset_path = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "WorldFeaturesAll")#ChlorWorldFeatures
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []

for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()

features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks

np.random.seed(42)

first_run = True

# Use tqdm for progress bar
for count, file in enumerate(tqdm(files, desc="Predicting on files")):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)

    # Perform multiple forward passes with dropout
    num_samples = 1  # Adjust this based on your requirements
    predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

    #with torch.no_grad():
    for i in range(num_samples):
        predictions = model.predict(features) #decoded, 

        # Store predictions for later averaging
        predictions_sum[i] = predictions.reshape(chunk_shape)

    # Average predictions over all samples
    predictions_avg = np.mean(predictions_sum, axis=0)
    predictions_var = np.var(predictions_sum, axis=0)

    if first_run:
        prediction_map2 = predictions_avg
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
        prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)


# ======================================================================
# Cell 31
# ======================================================================
print(features.shape)
print(X_mean.shape)


# ======================================================================
# Cell 32
# ======================================================================
#del features
gc.collect()
#"""    
land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1
#prediction_map2[land_map] = 0

#plt.figure(figsize=[25,15], dpi=200)
# import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
# plt.figure(figsize=[25, 15], dpi=50)
prediction_map2[land_map] = 0
# plt.imshow(prediction_map2 , norm=mcolors.LogNorm(vmin=1e-1, vmax=1e3), cmap="cubehelix")
# cbar = plt.colorbar()
#"""

np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / "prediction_map_MAR_entire_rf_n", prediction_map2)


# ======================================================================
# Cell 34
# ======================================================================
model_type = "xgboost"
n_estimators = 1000
model, training_mae, evaluation_mae, train_preds, test_preds = train_tree_based_model(
    train_features, train_labels, test_features, test_labels,
    model_type=model_type,  
    n_estimators=n_estimators
)


# ======================================================================
# Cell 35
# ======================================================================
plot_predictions(train_labels, train_preds, test_labels, test_preds, model_name="xgboost", n_estimators=1000)


# ======================================================================
# Cell 36
# ======================================================================
#os.chdir("/gxfs_work1/cau/sunms498/work/david/NN4SedRatePred/models/MLP/SedRate_MAR/models_torch/")



from os import walk  
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

#dataset_path = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "WorldFeaturesAll")#ChlorWorldFeatures
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []

for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()


features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks

np.random.seed(42)


first_run = True
ii = 0
features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks


first_run = True
ii = 0

for count, file in enumerate(files):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)


    # Perform multiple forward passes with dropout
    num_samples = 1  # Adjust this based on your requirements
    predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

    #with torch.no_grad():
    for i in range(num_samples):
        predictions = model.predict(features) #decoded, 

        # Store predictions for later averaging
        predictions_sum[i] = predictions.reshape(chunk_shape)

    # Average predictions over all samples
    predictions_avg = np.mean(predictions_sum, axis=0)
    predictions_var = np.var(predictions_sum, axis=0)

    if first_run:
        prediction_map2 = predictions_avg
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
        prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)

    ii += 1
    print("Prediction " + str(ii) + " done!")


# ======================================================================
# Cell 37
# ======================================================================
#del features
gc.collect()
#"""    
land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1
#prediction_map2[land_map] = 0

#plt.figure(figsize=[25,15], dpi=200)
# import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
# plt.figure(figsize=[25, 15], dpi=50)
prediction_map2[land_map] = 0
# plt.imshow(prediction_map2 , norm=mcolors.LogNorm(vmin=1e-1 , vmax=1e3), cmap="cubehelix")
# cbar = plt.colorbar()
#"""

np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / "prediction_map_MAR_entire_xgboost_n", prediction_map2)


# ======================================================================
# Cell 39
# ======================================================================
model_type = "catboost"
n_estimators = 1000
model, training_mae, evaluation_mae, train_preds, test_preds = train_tree_based_model(
    train_features, train_labels, test_features, test_labels,
    model_type=model_type, 
    n_estimators=n_estimators
)


# ======================================================================
# Cell 40
# ======================================================================
plot_predictions(train_labels, train_preds, test_labels, test_preds, model_name="catboost", n_estimators=1000)


# ======================================================================
# Cell 41
# ======================================================================
#os.chdir("/gxfs_work1/cau/sunms498/work/david/NN4SedRatePred/models/MLP/SedRate_MAR/models_torch/")



from os import walk  
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

#dataset_path = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "WorldFeaturesAll")#ChlorWorldFeatures
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []

for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()


features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks

np.random.seed(42)


first_run = True
ii = 0
features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks


first_run = True
ii = 0

for count, file in enumerate(files):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)


    # Perform multiple forward passes with dropout
    num_samples = 1  # Adjust this based on your requirements
    predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

    #with torch.no_grad():
    for i in range(num_samples):
        predictions = model.predict(features) #decoded, 

        # Store predictions for later averaging
        predictions_sum[i] = predictions.reshape(chunk_shape)

    # Average predictions over all samples
    predictions_avg = np.mean(predictions_sum, axis=0)
    predictions_var = np.var(predictions_sum, axis=0)

    if first_run:
        prediction_map2 = predictions_avg
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
        prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)

    ii += 1
    print("Prediction " + str(ii) + " done!")


# ======================================================================
# Cell 42
# ======================================================================
#del features
gc.collect()
#"""    
land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1
#prediction_map2[land_map] = 0

#plt.figure(figsize=[25,15], dpi=200)
# import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
# plt.figure(figsize=[25, 15], dpi=50)
prediction_map2[land_map] = 0
# plt.imshow(prediction_map2 , norm=mcolors.LogNorm(vmin=1e-1, vmax=1e3), cmap="cubehelix")
# cbar = plt.colorbar()
#"""

np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / "prediction_map_MAR_entire_catboost_n", prediction_map2)


# ======================================================================
# Cell 44
# ======================================================================
model, training_mae, evaluation_mae, train_preds, test_preds = train_knn(
        train_features, train_labels, test_features, test_labels,
        n_neighbors=9
    )


# ======================================================================
# Cell 45
# ======================================================================
def plot_predictions(y_train, train_preds, y_test, test_preds, model_name, n_estimators):
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0, 1]
    test_corr = np.corrcoef(y_test, test_preds)[0, 1]

    # Combine train and test for density plot
    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([train_preds, test_preds])

    # Find the overall min and max for axis and binning
    min_lim = min(all_true.min(), all_pred.min())
    max_lim = max(all_true.max(), all_pred.max())

    # Clamp display axis at 1500 as maximum
    min_display = min_lim
    max_display = min(1500, max_lim)

    # plt.figure(figsize=(7, 7))

    # Use 2D histogram for density, limiting counts per bin to max 15
    # counts, xedges, yedges, im = plt.hist2d(
        # all_true, all_pred,
        # bins=100,
        # range=[[min_display, max_display], [min_display, max_display]],
        # cmap='viridis',
        # cmin=0, vmax=15
    )
    # Set up colorbar and annotate with ">15" for max value
    # cbar = plt.colorbar(im)
    # cbar.set_label('Counts', fontsize=12)
    
    # Generate the colorbar ticks up to 15
    cbar_ticks = np.arange(0, 16, 3)
    if cbar_ticks[-1] != 15:
        cbar_ticks = np.append(cbar_ticks, 15)
    cbar.set_ticks(cbar_ticks)
    ticklabels = [f"{int(t)}" for t in cbar_ticks]
    if len(ticklabels) > 0:
        ticklabels[-1] = ">15"
    cbar.set_ticklabels(ticklabels)

    # Set specified grid values at 300, 600, 1200, 1500
    grid_values = [300, 600, 900, 1200, 1500]
    # plt.xlim(min_display, max_display)
    # plt.ylim(min_display, max_display)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().set_xticks(grid_values)
    # plt.gca().set_yticks(grid_values)

    # plt.plot([min_display, max_display], [min_display, max_display], 'k--', lw=2)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    #plt.title(f"{model_name} Predictions (n_estimators={n_estimators:.0f})")

    # Place the statistics in a textbox - LEFT TOP CORNER (x,y = 0.05, 0.85)
    textstr = (
        f"Train MAE: {train_mae:.3f}, CC: {train_corr:.3f}\n"
        f"Test MAE: {test_mae:.3f}, CC: {test_corr:.3f}"
    )
    # plt.gcf().text(0.15, 0.80, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5), ha='left', va='top')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


# ======================================================================
# Cell 46
# ======================================================================
plot_predictions(
    train_labels, train_preds,
    test_labels, test_preds,
    model_name="KNN",
    n_estimators=9  # Here, we pass k instead of n_estimators
)


# ======================================================================
# Cell 47
# ======================================================================
#os.chdir("/gxfs_work1/cau/sunms498/work/david/NN4SedRatePred/models/MLP/SedRate_MAR/models_torch/")


from os import walk  
feature_path = Path(Path() / "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men/")
X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

#dataset_path = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "WorldFeaturesAll")#ChlorWorldFeatures
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []

for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break

files.sort()


features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks

np.random.seed(42)


first_run = True
ii = 0
features = np.load(Path(dataset_path / files[0]))

chunk_shape = [6,4320] #360 chunks


first_run = True
ii = 0

for count, file in enumerate(files):
    features = np.load(Path(dataset_path / file))
    features = np.divide((features - X_mean), X_std)


    # Perform multiple forward passes with dropout
    num_samples = 1  # Adjust this based on your requirements
    predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

    #with torch.no_grad():
    for i in range(num_samples):
        predictions = model.predict(features) #decoded, 

        # Store predictions for later averaging
        predictions_sum[i] = predictions.reshape(chunk_shape)

    # Average predictions over all samples
    predictions_avg = np.mean(predictions_sum, axis=0)
    predictions_var = np.var(predictions_sum, axis=0)

    if first_run:
        prediction_map2 = predictions_avg
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_avg, axis=0)
        prediction_map2_var = np.append(prediction_map2_var, predictions_var, axis=0)

    ii += 1
    print("Prediction " + str(ii) + " done!")


# ======================================================================
# Cell 48
# ======================================================================
#del features
gc.collect()
#"""    
land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1
#prediction_map2[land_map] = 0

#plt.figure(figsize=[25,15], dpi=200)
# import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
# plt.figure(figsize=[25, 15], dpi=50)
#prediction_map2[land_map] = 0
# plt.imshow(prediction_map2 , vmin = 0, vmax = 1000, cmap="cubehelix")
# cbar = plt.colorbar()
#"""


# ======================================================================
# Cell 49
# ======================================================================

np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / "prediction_map_MAR_entire_knn_n", prediction_map2)

