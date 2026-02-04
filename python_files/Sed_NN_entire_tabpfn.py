# Standard library imports
from pathlib import Path
import sys
import random

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
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

set_seed()

# Load world features



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


from sklearn.model_selection import StratifiedKFold


test_size_ratio = 0.25
print("Test size ratio") 
print(test_size_ratio)

# Create bins for stratification
n_bins = 10
y_binned = pd.qcut(labels_DO, q=n_bins, labels=False, duplicates='drop')
train_features_DO, train_labels_DO, test_features_DO, test_labels_DO = split_train_test_data(features_DO, labels_DO, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)


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


features_CS, labels_CS, lats_CS, lons_CS = preprocess_features_labels(features, labels, lats, lons, X_mean, X_std)


from sklearn.model_selection import StratifiedKFold


test_size_ratio = 0.25
print("Test size ratio") 
print(test_size_ratio)

# Create bins for stratification
n_bins = 10
y_binned = pd.qcut(labels_CS, q=n_bins, labels=False, duplicates='drop')
train_features_CS, train_labels_CS, test_features_CS, test_labels_CS = split_train_test_data(features_CS, labels_CS, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)

train_features = np.concatenate([train_features_CS, train_features_DO], axis = 0)
test_features = np.concatenate([test_features_CS, test_features_DO], axis = 0)
train_labels = np.concatenate([train_labels_CS, train_labels_DO], axis = 0)
test_labels = np.concatenate([test_labels_CS, test_labels_DO], axis = 0)
features = np.concatenate([features_CS, features_DO], axis = 0)
labels = np.concatenate([labels_CS, labels_DO], axis = 0)


# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
from tabpfn import TabPFNRegressor  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load Boston Housing data
from sklearn.datasets import make_regression
import pandas as pd


# Initialize the regressor
regressor = TabPFNRegressor(ignore_pretraining_limits=True)  
regressor.fit(train_features, train_labels)

# Predict on the test set
predictions = regressor.predict(test_features)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

print("Mean Squared Error (MSE):", mse)

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

"""
# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot for training data
plt.scatter(train_labels, regressor.predict(train_features), color='blue', label='Train', alpha=0.7)
# Scatter plot for test data
plt.scatter(test_labels, predictions, color='red', label='Test', alpha=0.7)

# Calculate and display the correlation coefficient
correlation = np.corrcoef(test_labels, predictions)[0, 1]
plt.text(0.1, 0.9, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12)

plt.title('TabPFN: Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
"""
from os import walk  
feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_CS_men"
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


        # Use output_type="full" to get all outputs, including quantiles and mean
        result = regressor.predict(features, output_type="full", quantiles=[0.05, 0.5, 0.95])
        predictions_mean = result["mean"].reshape(chunk_shape)
        quantiles_dict = result["quantiles"]
        predictions_lower = quantiles_dict[0].reshape(chunk_shape)
        predictions_median = quantiles_dict[1].reshape(chunk_shape)
        predictions_upper = quantiles_dict[2].reshape(chunk_shape)
        predictions_var = ((predictions_upper - predictions_lower) / 2) ** 2  # proxy for variance



    if first_run:
        prediction_map2 = predictions_mean
        prediction_map2_var = predictions_var
        first_run = False
    else:
        prediction_map2 = np.append(prediction_map2, predictions_mean, axis=0)
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
import matplotlib.colors as mcolors
prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
plt.figure(figsize=[25, 15], dpi=50)
#prediction_map2[land_map] = 0
#plt.imshow(prediction_map2 , vmin = 0, vmax = 1000, cmap="cubehelix")
#cbar = plt.colorbar()
#"""

np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / "prediction_map_SedRate_entire_tabpfn", prediction_map2)