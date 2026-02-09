"""
Hyperparameter optimization for supervised neural network models.

This script performs hyperparameter optimization for neural network models
predicting sediment accumulation rates. It tests various network architectures
and saves the best performing models and their prediction maps.
"""

import warnings
warnings.filterwarnings('ignore')

# Standard library imports
import sys
import random
from pathlib import Path
from os import walk

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project paths
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import PROJECT_ROOT, DATA_INTERIM, DATA_RAW, DATA_OUTPUT

models_path = project_root / "src" / "models"
postproc_path = project_root / "src" / "postprocessing"
sys.path.extend([str(models_path), str(postproc_path)])

# Import project modules
from train_neural_network import (
    preprocess_features_labels, split_train_test_data, 
    create_dataloaders, buildDNN, set_seed
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configure PyTorch settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

set_seed(42)

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

import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_constraint):
        super(CombinedLoss, self).__init__()
        self.lambda_constraint = lambda_constraint
        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss()

    def forward(self, features, labels, decoded, targets, porosity, train_step):


        constraint_loss = torch.mean(torch.clamp(targets * (porosity - 0.6), min=0))  # Hinge loss
        
        if train_step == 1:
            features = torch.reshape(features, (features.shape[0]*features.shape[1],1))
            decoded = torch.reshape(decoded, (decoded.shape[0]*decoded.shape[1],1))
            reconstruction_loss = self.MSELoss(features, decoded)  # Example: using L1 loss
            combined_loss = reconstruction_loss + self.lambda_constraint * constraint_loss
            
        elif train_step == 2:
            labels = torch.reshape(labels, (labels.shape[0],1))
            targets = torch.reshape(targets, (targets.shape[0],1))
            regression_loss = self.L1Loss(labels, targets)
            combined_loss = regression_loss + self.lambda_constraint * constraint_loss
            
        elif train_step == 3:
            features = torch.reshape(features, (features.shape[0]*features.shape[1],1))
            decoded = torch.reshape(decoded, (decoded.shape[0]*decoded.shape[1],1))
            labels = torch.reshape(labels, (labels.shape[0],1))
            targets = torch.reshape(targets, (targets.shape[0],1))
            regression_loss = self.L1Loss(labels, targets)
            reconstruction_loss = self.MSELoss(features, decoded)  # Example: using L1 loss
            combined_loss = reconstruction_loss + regression_loss + self.lambda_constraint * constraint_loss
            
        else:
            raise ValueError("Invalid train_step value.")
         
        return combined_loss

#loss_fn = CombinedLoss(lambda_constraint=0.5)#
#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.HuberLoss()



import torch.nn as nn

class FEA(nn.Module):
    def __init__(self, layer_width):
        super(FEA, self).__init__()

        self.do_prob = 0.2
        self.encoder = nn.Sequential()

        # Dynamically create encoder layers
        self.encoder.add_module(f'encoder_linear_0', nn.Linear(features.shape[1], layer_width[0]))
        #self.encoder.add_module(f'encoder_batchnorm_0', nn.BatchNorm1d(layer_width[0]))
        self.encoder.add_module(f'encoder_relu_0', nn.ReLU())
        self.encoder.add_module(f'encoder_dropout_0', nn.Dropout(self.do_prob))

        for i in range(1, len(layer_width)): # start from the second layer, index 1
            self.encoder.add_module(f'encoder_linear_{i}', nn.Linear(layer_width[i-1], layer_width[i])) # corrected input size
            #self.encoder.add_module(f'encoder_batchnorm_{i}', nn.BatchNorm1d(layer_width[i]))
            self.encoder.add_module(f'encoder_relu_{i}', nn.ReLU())
            self.encoder.add_module(f'encoder_dropout_{i}', nn.Dropout(self.do_prob))

        #self.decoder = nn.Sequential()

        # Dynamically create decoder layers (reverse of encoder)
        """
        for i in range(len(layer_width) - 1, 0, -1):
            if i == len(layer_width) - 1:
                self.decoder.add_module(f'decoder_linear_{i}', nn.Linear(layer_width[i], layer_width[i - 1]))
            else:
                self.decoder.add_module(f'decoder_linear_{i}', nn.Linear(layer_width[i + 1], layer_width[i - 1]))
            self.decoder.add_module(f'decoder_batchnorm_{i}', nn.BatchNorm1d(layer_width[i - 1]))
            self.decoder.add_module(f'decoder_relu_{i}', nn.ReLU())
            self.decoder.add_module(f'decoder_dropout_{i}', nn.Dropout(self.do_prob))

        self.decoder.add_module('decoder_output', nn.Linear(layer_width[0], features.shape[1]))
        """
        self.supervised = nn.Sequential(
            nn.Linear(layer_width[-1], 1),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        targets = self.supervised(encoded)
        return decoded, targets#decoded, 

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)



# In[669]:



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


features_DO, labels_DO, lats_DO, lons_DO = preprocess_features_labels(
    features, labels, lats, lons, X_mean, X_std
)

# Split data for training and testing
test_size_ratio = 0.25
print(f"Test size ratio: {test_size_ratio}")

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


features_CS, labels_CS, lats_CS, lons_CS = preprocess_features_labels(
    features, labels, lats, lons, X_mean, X_std
)

# Split CS data for training and testing
test_size_ratio = 0.25
print(f"Test size ratio: {test_size_ratio}")

# Create bins for stratification
n_bins = 10
y_binned = pd.qcut(labels_CS, q=n_bins, labels=False, duplicates='drop')
train_features_CS, train_labels_CS, test_features_CS, test_labels_CS = split_train_test_data(features_CS, labels_CS, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)

# Combine CS and DO data
train_features = np.concatenate([train_features_CS, train_features_DO], axis=0)
test_features = np.concatenate([test_features_CS, test_features_DO], axis=0)
train_labels = np.concatenate([train_labels_CS, train_labels_DO], axis=0)
test_labels = np.concatenate([test_labels_CS, test_labels_DO], axis=0)
features = np.concatenate([features_CS, features_DO], axis=0)
labels = np.concatenate([labels_CS, labels_DO], axis=0)

# Create data loaders
batch_size = 64
trainloader, testloader = create_dataloaders(
    train_features, train_labels, test_features, test_labels, 
    batch_size=batch_size, shuffle_train=True, shuffle_test=False
)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_predictions():
    """Generate global prediction map using Monte Carlo dropout."""
    # Load normalization parameters
    feature_path = DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_CS_men"
    X_mean = np.load(Path(feature_path / "features_mean.npy")).astype(float)
    X_std = np.load(Path(feature_path / "features_std.npy")).astype(float)

    # Load world features
    dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
    files = []

    for (dirpath, dirnames, filenames) in walk(dataset_path):
        files.extend(filenames)
        break

    files.sort()


    # Set model to evaluation mode with dropout enabled for MC dropout
    model.eval()
    enable_dropout(model)
    
    chunk_shape = [6, 4320]  # 360 chunks
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Move model to device
    model.to(device)
    model.double()
    
    first_run = True

    for count, file in enumerate(files):
        features = np.load(Path(dataset_path / file))
        features = np.divide((features - X_mean), X_std)

        features = torch.tensor(features)
        features = features.to(device)

        # Perform multiple forward passes with dropout (Monte Carlo dropout)
        num_samples = 100
        predictions_sum = np.zeros((num_samples, chunk_shape[0], chunk_shape[1]))

        with torch.no_grad():
            for i in range(num_samples):
                predictions = model(features)
                predictions_sum[i] = predictions.cpu().detach().numpy().reshape(chunk_shape)

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
        if ii == 359:
            print("Prediction " + str(ii) + " done!")

    #del features
    gc.collect()
    """    
    land_file = DATA_RAW / "island_map.npy"
    land_map = np.load(land_file)
    land_map[np.isnan(land_map)] = 1
    #prediction_map2[land_map] = 0

    #plt.figure(figsize=[25,15], dpi=200)
    import matplotlib.colors as mcolors
    #prediction_map2 = np.rot90(np.rot90(np.fliplr(prediction_map2)))
    plt.figure(figsize=[25, 15], dpi=50)
    #prediction_map2[land_map] = 0
    plt.imshow(prediction_map2, norm=mcolors.LogNorm(vmin=1e1, vmax=1e2),cmap="cubehelix")
    cbar = plt.colorbar()
    """


    np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / f"prediction_map_SedRate_entire_supervised_{layer_widths}", prediction_map2)


train_step = 2

layer_widths_all = [[2]*3,
                     [4]*3,
                     [2**3, 2**2, 2],
                     [2**5, 2**4, 2**3],
                     [2**7, 2**6, 2**5], 
                     [2**9, 2**8, 2**7], 
                     [128]*3, 
                     [256]*3, 
                     [512]*3,
                    
                     #4layers
                     [16, 2**3, 2**2, 2], 
                     [32, 2**4, 2**3, 4],
                    [2**9, 2**8, 2**7, 64],
                     [128]*4,
                     [256]*4,
                     [512]*4,
                    
                     #5 layers
                     [32, 2**4, 2**3, 2**2, 2],
                     [2**8, 2**7, 2**6, 2**5, 16],
                     [2**9, 2**8, 2**7, 2**6, 32],
                     [32]*5,
                     [64]*5,
                     [128]*5,
                     [256]*5,
                     [512]*5,
                    
                     #6 layers
                     [2**9, 2**8, 2**7, 2**6, 2**5, 16],
                     [2**8, 2**7, 2**6, 2**5, 2**4, 8],
                     [128]*6,
                     [64]*6,
                     [32]*6,
                     [256]*6,
                     [512]*6,
                    
                     #7 layers
                     [256]*7,
                     [512]*7,
                     [2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 8],
                     [2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 4],
                    
                     #8 layers
                     [256]*8,
                     [512]*8,
                     [2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 4],
                     [2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2],
                    
                     #9 layers
                     [256]*9,
                     [512]*9,
                     [2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2],
                    
                    
                    
                     #10 layers
                     [2**10, 2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2],
                     [64]*10,
                     [128]*10,
                     [256]*10,
                     [512]*10,
                    ]
train_step = 2

#layer_widths_all = [[128]*10]
for layer_widths in layer_widths_all:
    set_seed(42)
    # Example usage:
    input_dim = features.shape[1]  # Replace with the actual input dimension
    # Define your desired layer widthsmodel = FEA(layer_widths)#FEA(input_dim, layer_widths)
    model = buildDNN(layer_widths, features)
    model.apply(initialize_weights)
    model.double()


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-3,
                                 weight_decay = 1e-8)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, 
                                                              patience=50, threshold=0.0001, threshold_mode='rel', 
                                                              cooldown=0, min_lr=0.00001, eps=1e-08)#, verbose=False)


    reconstruction_loss_factor = 0
    target_loss_factor = 1
    count = 1



    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

            
    num_epochs = 1800

    training_accuracy, training_losses, training_mae, learning_rate, evaluation_losses, evaluation_mae, true_labels, pred_labels =  train_model(model, loss_fn, trainloader, testloader, num_epochs = num_epochs)

    features = torch.tensor(features, dtype=torch.float64)
    labels = torch.tensor(labels, dtype=torch.float64)

    # Then move to GPU if available
    if torch.cuda.is_available():
        features = features.cuda()
        labels = labels.cuda()

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("layer widths")
    print(layer_widths)
    print("evaluation losses")
    print(evaluation_mae[-5:])
    print("training loses")
    print(training_mae[-5:])
    output_path = "test_img"
    output_path = str(output_path)
    print("+++++++++test results+++++++++")
    test_labels, test_targets = evaluate_model(model, testloader, output_path)
    print("+++++++++all results+++++++++")
    all_labels, all_targets =  evaluate_all_predictions(model, features, labels, output_path)
    print("+++++++++train results+++++++++")
    train_labels, train_targets = evaluate_model(model, trainloader, output_path)    

    get_predictions()



