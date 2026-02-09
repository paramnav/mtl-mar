"""
Hyperparameter optimization for multi-task learning (MTL) models.

This script performs hyperparameter optimization for multi-task learning models
that simultaneously predict sediment accumulation rates (MAR) and total organic
carbon (TOC). It tests various network architectures and shared layer configurations.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).resolve().parents[1]
models_path = project_root / "src" / "models"
postproc_path = project_root / "src" / "postprocessing"
sys.path.insert(0, str(project_root / "src"))
sys.path.extend([str(models_path), str(postproc_path)])

from utils import PROJECT_ROOT, DATA_INTERIM, DATA_RAW, DATA_OUTPUT

# Import project modules
from train_multi_task_learning import (
    preprocess_features_labels, split_train_test_data, set_seed
)
from predict_neural_network import *
from evaluate_model_performance import *

# Set random seed for reproducibility
set_seed(42)

# Set device and torch backend options
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameter search space
constraint = 0.1
layer_widths_all = [
    [64]*3, [64]*4, [512]*3, [512]*4, [512]*5, 
    [512, 256, 128], [512, 256, 128, 64], [512, 256, 128, 64, 32]
]
num_shared_layers_all = [1, 2, 3, 4, 5]

# Load TOC (Total Organic Carbon) data
# Note: TOC features from nn-toc-update project - adjust path if needed
toc_data_path = project_root.parent / "nn-toc-update" / "notebooks" / "TOC" / "preprocessed"
if not toc_data_path.exists():
    toc_data_path = DATA_INTERIM / "toc_preprocessed"
features_toc_CS = np.load(toc_data_path / "features_CS_toc.npy")
labels_toc_CS = np.load(toc_data_path / "labels_CS_toc.npy")
labels_mean_toc_CS = np.mean(labels_toc_CS)
labels_std_toc_CS = np.std(labels_toc_CS)

n_bins = 10
y_binned = pd.qcut(labels_toc_CS, q=n_bins, labels=False, duplicates='drop')
train_features_toc_CS, train_labels_toc_CS, test_features_toc_CS, test_labels_toc_CS = split_train_test_data(features_toc_CS, labels_toc_CS, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)


features_toc_DO = np.load(toc_data_path / "features_DO_toc.npy")
labels_toc_DO = np.load(toc_data_path / "labels_DO_toc.npy")
labels_mean_toc_DO = np.mean(labels_toc_DO)
labels_std_toc_DO = np.std(labels_toc_DO)


y_binned = pd.qcut(labels_toc_DO, q=n_bins, labels=False, duplicates='drop')
train_features_toc_DO, train_labels_toc_DO, test_features_toc_DO, test_labels_toc_DO = split_train_test_data(features_toc_DO, labels_toc_DO, test_size_ratio=0.25, random_seed=42, stratify=y_binned)
#trainloader, testloader = create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)


toc_train_features = np.concatenate([train_features_toc_CS, train_features_toc_DO], axis = 0)
toc_test_features = np.concatenate([test_features_toc_CS, test_features_toc_DO], axis = 0)
toc_train_labels = np.concatenate([train_labels_toc_CS, train_labels_toc_DO], axis = 0)
toc_test_labels = np.concatenate([test_labels_toc_CS, test_labels_toc_DO], axis = 0)

features_toc = np.concatenate([features_toc_CS, features_toc_DO], axis = 0)
labels_toc = np.concatenate([labels_toc_CS, labels_toc_DO], axis = 0)



# Load world features
dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = sorted([f for f in dataset_path.iterdir() if f.is_file()])
features = np.load(files[0])
world_feature_shape_dim = features.shape[1]

#MASS ACCUMULATION RATES
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

sedrate_train_features = np.concatenate([train_features_CS, train_features_DO], axis = 0)
sedrate_test_features = np.concatenate([test_features_CS, test_features_DO], axis = 0)
sedrate_train_labels = np.concatenate([train_labels_CS, train_labels_DO], axis = 0)
sedrate_test_labels = np.concatenate([test_labels_CS, test_labels_DO], axis = 0)
sedrate_features = np.concatenate([features_CS, features_DO], axis = 0)
sedrate_labels = np.concatenate([labels_CS, labels_DO], axis = 0)
#batch_size = 64
#trainloader, testloader = create_dataloaders(sedrate_train_features, sedrate_train_labels, sedrate_test_features, sedrate_test_labels, batch_size=batch_size, shuffle_train=True, shuffle_test=False)




dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
files = []
for (dirpath, dirnames, filenames) in walk(dataset_path):
    files.extend(filenames)
    break
files.sort()
features = np.load(Path(dataset_path / files[0]))

print("features_toc shape:", features_toc.shape)
print("features_sedrate shape:", sedrate_features.shape)
if features_toc.shape[1] == sedrate_features.shape[1]:
    print("The dimension of features_toc and features_sedrate are equal.")
else:
    print(f"The dimension is not equal: features_toc.shape[1]={features_toc.shape[1]}, sedrate_features.shape[1]={sedrate_features.shape[1]}")




# %%

# Imparting constraints in the loss function. when porosity is greater than 0.6, sed rate is equal to zero.

import torch.nn.functional as F

class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_constraint):
        super(CombinedLoss, self).__init__()
        self.lambda_constraint = lambda_constraint
        self.loss_fn = {
            'sedimentation_rates': torch.nn.HuberLoss(),
            'toc': torch.nn.HuberLoss(),
        }
        self.MSELoss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss()

    def forward(self, features, labels, targets, decoded, porosity, train_step, task_id):


        fine_sand_porosity = -1.3026080219130773 #(0.55-X_mean)/X_std

        constraint_loss = torch.mean(torch.clamp(targets**2 * (fine_sand_porosity -porosity), min=0))  # Hinge loss
        mse_constant = torch.mean(torch.max(porosity, torch.tensor(fine_sand_porosity, device=porosity.device)) / torch.mean(porosity))
        
        if train_step == 1:
            features = torch.reshape(features, (features.shape[0]*features.shape[1],1))
            decoded = torch.reshape(decoded, (decoded.shape[0]*decoded.shape[1],1))
            reconstruction_loss = self.MSELoss(features, decoded)  # Example: using L1 loss
            combined_loss = reconstruction_loss# + self.lambda_constraint * constraint_loss
            
        elif train_step == 2:
            labels = torch.reshape(labels, (labels.shape[0],1))
            targets = torch.reshape(targets, (targets.shape[0],1))
            if task_id == "sedimentation_rates":
                regression_loss = self.loss_fn["sedimentation_rates"](labels, targets)
            if task_id == "toc":
                regression_loss = self.loss_fn["toc"](labels, targets)
            #combined_loss = mse_constant*regression_loss + self.lambda_constraint * constraint_loss #no constraint for deep sea
            combined_loss = regression_loss
            
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

#loss_fn = CombinedLoss(lambda_constraint=constraint)#
loss_fn = torch.nn.HuberLoss()
"""
loss_fn = {
    'sedimentation_rates': torch.nn.L1Loss(),
    'toc': torch.nn.MSELoss(),
}
"""

def evaluate_model(model, data_loader, task, device):
    all_labels = []
    all_outputs = []

    model.eval()
    with torch.no_grad():
        for features, labels in data_loader:
            features = torch.tensor(features).to(device)
            features = features.to(dtype=torch.double)
            labels = labels.reshape(-1)

            decoded, outputs = model(features, task)
            outputs = outputs.cpu().numpy().reshape(-1)

            all_labels.extend(labels.cpu().detach().numpy())
            all_outputs.extend(outputs)
    #all_labels = all_labels.cpu().detach().numpy()
    all_labels = np.array(all_labels)
    #all_labels = all_labels.cpu().detach().numpy()
    all_outputs = np.array(all_outputs)

    mae = mean_absolute_error(all_labels, all_outputs)
    mse = mean_squared_error(all_labels, all_outputs)
    correlation = np.corrcoef(all_labels, all_outputs)[0, 1]
    r2 = r2_score(all_labels, all_outputs)

    return mae, mse, correlation, r2




# %%

count_mtl = 0
for layer_width in layer_widths_all:
    for num_shared_layers in num_shared_layers_all:
        
        print(f"Training MTL model {count_mtl} with layer_widths={layer_width} and num_shared_layers={num_shared_layers}")
        count_mtl = count_mtl + 1
        if count_mtl<25:
            continue
        layer_widths = layer_width
        num_shared_layers = num_shared_layers
        if num_shared_layers > len(layer_widths):
            continue

        print(sedrate_train_features.shape)
        sedrate_train_features = torch.tensor(sedrate_train_features, dtype=torch.float32)
        sedrate_train_labels = torch.tensor(sedrate_train_labels, dtype=torch.float32)
        sedrate_test_features = torch.tensor(sedrate_test_features, dtype=torch.float32)
        sedrate_test_labels = torch.tensor(sedrate_test_labels, dtype=torch.float32)

        toc_train_features = torch.tensor(toc_train_features, dtype=torch.float32)
        toc_train_labels = torch.tensor(toc_train_labels, dtype=torch.float32)
        toc_test_features = torch.tensor(toc_test_features, dtype=torch.float32)
        toc_test_labels = torch.tensor(toc_test_labels, dtype=torch.float32)
        # Create TensorDatasets
        sedrate_train_dataset = TensorDataset(sedrate_train_features, sedrate_train_labels)
        sedrate_test_dataset = TensorDataset(sedrate_test_features, sedrate_test_labels)
        toc_train_dataset = TensorDataset(toc_train_features, toc_train_labels)
        toc_test_dataset = TensorDataset(toc_test_features, toc_test_labels)
        

        # Create DataLoaders
        batch_size = 32
        train_dataloaders = {
            'sedimentation_rates': DataLoader(sedrate_train_dataset, batch_size=batch_size, shuffle=True),
            'toc': DataLoader(toc_train_dataset, batch_size=batch_size, shuffle=True)
        }
        test_dataloaders = {
            'sedimentation_rates': DataLoader(sedrate_test_dataset, batch_size=batch_size, shuffle=False),
            'toc': DataLoader(toc_test_dataset, batch_size=batch_size, shuffle=False)
        }

        # Now you can use train_dataloaders and test_dataloaders in your training loop, e.g.:
        # for features, labels in train_dataloaders['sedimentation_rates']:
        #     # training step here

        # %%
        print("Train Labels Distribution(Sedimentation Rates):")
        print(f"Count: {len(sedrate_train_labels)}")
        print(f"Mean: {sedrate_train_labels.mean():.3f}")
        print(f"Std: {sedrate_train_labels.std():.3f}")
        print(f"Min: {sedrate_train_labels.min():.3f}")
        print(f"Max: {sedrate_train_labels.max():.3f}")
        print("\nTest Labels Distribution(Sedimentation Rates):")
        print(f"Count: {len(sedrate_test_labels)}")
        print(f"Mean: {sedrate_test_labels.mean():.3f}") 
        print(f"Std: {sedrate_test_labels.std():.3f}")
        print(f"Min: {sedrate_test_labels.min():.3f}")
        print(f"Max: {sedrate_test_labels.max():.3f}")

        # Plot histograms
        import matplotlib.pyplot as plt

        # Count number of zeros in train and test labels
        train_zeros = (sedrate_train_labels == 0).sum()
        test_zeros = (sedrate_test_labels == 0).sum()
        print(f"Number of zeros in train labels: {train_zeros}")
        print(f"Number of zeros in test labels: {test_zeros}")

        plt.figure(figsize=(10,5))
        from scipy.stats import gaussian_kde
        plt.subplot(1,2,1)
        plt.hist(sedrate_train_labels, bins=50, alpha=0.5, density=True)
        kde = gaussian_kde(sedrate_train_labels)
        x_range = np.linspace(min(sedrate_train_labels), max(sedrate_train_labels), 200)
        plt.plot(x_range, kde(x_range), alpha=0.7)
        plt.title('Train Labels Distribution')
        plt.xlabel('Label Value')
        plt.ylabel('Count')

        plt.subplot(1,2,2)
        plt.hist(sedrate_test_labels, bins=50, alpha=0.5, density=True)
        kde = gaussian_kde(sedrate_test_labels)
        x_range = np.linspace(min(sedrate_test_labels), max(sedrate_test_labels), 200)
        plt.plot(x_range, kde(x_range), alpha=0.7)
        plt.title('Test Labels Distribution')
        plt.xlabel('Label Value')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()


        # %%
        print("Train Labels Distribution(TOC):")
        print(f"Count: {len(toc_train_labels)}")
        print(f"Mean: {toc_train_labels.mean():.3f}")
        print(f"Std: {toc_train_labels.std():.3f}")
        print(f"Min: {toc_train_labels.min():.3f}")
        print(f"Max: {toc_train_labels.max():.3f}")
        print("\nTest Labels Distribution(TOC):")
        print(f"Count: {len(toc_test_labels)}")
        print(f"Mean: {toc_test_labels.mean():.3f}") 
        print(f"Std: {toc_test_labels.std():.3f}")
        print(f"Min: {toc_test_labels.min():.3f}")
        print(f"Max: {toc_test_labels.max():.3f}")

        # Plot histograms
        import matplotlib.pyplot as plt

        # Count number of zeros in train and test labels
        train_zeros = (toc_train_labels == 0).sum()
        test_zeros = (toc_test_labels == 0).sum()
        print(f"Number of zeros in train labels: {train_zeros}")
        print(f"Number of zeros in test labels: {test_zeros}")

        plt.figure(figsize=(10,5))
        from scipy.stats import gaussian_kde
        plt.subplot(1,2,1)
        plt.hist(toc_train_labels, bins=50, alpha=0.5, density=True)
        kde = gaussian_kde(toc_train_labels)
        x_range = np.linspace(min(toc_train_labels), max(toc_train_labels), 200)
        plt.plot(x_range, kde(x_range), alpha=0.7)
        plt.title('Train Labels Distribution')
        plt.xlabel('Label Value')
        plt.ylabel('Count')

        plt.subplot(1,2,2)
        plt.hist(toc_test_labels, bins=50, alpha=0.5, density=True)
        kde = gaussian_kde(toc_test_labels)
        x_range = np.linspace(min(toc_test_labels), max(toc_test_labels), 200)
        plt.plot(x_range, kde(x_range), alpha=0.7)
        plt.title('Test Labels Distribution')
        plt.xlabel('Label Value')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()


        # %%
        toc_test_features.shape

        # %%
        """
        # Convert all arrays to PyTorch tensors (if not already)
        toc_features_train = torch.tensor(toc_features_train, dtype=torch.float32)
        toc_labels_train = torch.tensor(toc_labels_train, dtype=torch.float32)

        sedrate_features_train = torch.tensor(sedrate_features_train, dtype=torch.float32)
        sedrate_labels_train = torch.tensor(sedrate_labels_train, dtype=torch.float32)

        toc_features_test = torch.tensor(toc_features_test, dtype=torch.float32)
        toc_labels_test = torch.tensor(toc_labels_test, dtype=torch.float32)

        sedrate_features_test = torch.tensor(sedrate_features_test, dtype=torch.float32)
        sedrate_labels_test = torch.tensor(sedrate_labels_test, dtype=torch.float32)

        # Now wrap into DataLoaders
        train_data = {
            'toc': DataLoader(TensorDataset(toc_features_train, toc_labels_train), batch_size=64, shuffle=True),
            'sedimentation_rates': DataLoader(TensorDataset(sedrate_features_train, sedrate_labels_train), batch_size=64, shuffle=True)
        }

        test_data = {
            'toc': DataLoader(TensorDataset(toc_features_test, toc_labels_test), batch_size=64, shuffle=False),
            'sedimentation_rates': DataLoader(TensorDataset(sedrate_features_test, sedrate_labels_test), batch_size=64, shuffle=False)
        }
        """



        # %%
        #test_data['sedimentation_rates'][0][1].shape

        # %%
        features_list = []
        labels_list = []
        for task in train_dataloaders:
            if task == "toc":
                continue
            for features, labels in train_dataloaders[task]:
                features_list.append(features)
                labels_list.append(labels)


        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        print(features.shape)
        print(labels.shape)

        # %%
        learning_rates = []  # List to store learning rates
        train_errors_task1 = []  # List to store training errors for task 1
        train_errors_task2 = []  # List to store training errors for task 2
        test_errors_task1 = []  # List to store testing errors for task 1
        test_errors_task2 = []  # List to store testing errors for task 2

        train_step = 2
        #"""
        num_epochs = 2000
        model = buildDNN(layer_widths, sedrate_features, num_shared_layers)
        print(count_mtl)
        print(model)
        print(num_epochs)
        print(layer_widths)
        print(num_shared_layers)
        #for param in model.decoder.parameters():
        #    param.requires_grad = False    
            #for key in loss_fn:
            #    loss_fn[key] = loss_fn[key].cuda()
        #"""
        if torch.cuda.is_available():
            model = model.cuda()    
            loss_fn = loss_fn.cuda()

        print("Task 1 : Sedimentation rates")
        print("Task 2 : TOC")

        # Training loop
        training_losses, learning_rate, evaluation_losses =  train_model(model, loss_fn, train_dataloaders, test_dataloaders, train_step=2, num_epochs=num_epochs)
            

        # %%
        from sklearn.metrics import r2_score        
        # Plotting the learning rate for each task
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rate, label='Learning Rate - Task 1')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate - Task 1')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(learning_rate, label='Learning Rate - Task 2')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate - Task 2')
        plt.legend()
        plt.show()

        # Plotting the train errors for each task
        plt.figure(figsize=(10, 6))
        training_losses_mar = [training_loss['sedimentation_rates'] for training_loss in training_losses]
        plt.plot(training_losses_mar, label='Training Error - MAR')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training Error - Task 1')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        training_losses_toc = [training_loss['sedimentation_rates'] for training_loss in training_losses]
        plt.plot(training_losses_toc, label='Training Error - TOC')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training Error - TOC')
        plt.legend()
        plt.show()

        # Plotting the test errors for each task
        plt.figure(figsize=(10, 6))
        testing_losses_mar = [testing_loss['sedimentation_rates'] for testing_loss in evaluation_losses]
        plt.plot(testing_losses_mar, label='Testing Error - MAR')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Testing Error - Task 1')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        testing_losses_toc = [testing_loss['toc'] for testing_loss in evaluation_losses]
        plt.plot(testing_losses_toc, label='Testing Error - TOC')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Testing Error - TOC')
        plt.legend()
        plt.show()

        for task in test_dataloaders:
            features_list = []
            labels_list = []
            for features, labels in test_dataloaders[task]:
                features = torch.tensor(features).to(device)
                features = features.to(dtype=torch.double)
                labels = torch.tensor(labels).to(device)
                labels = labels.to(dtype=torch.double)
                features_list.append(features)
                labels_list.append(labels)
            # After the loop, concatenate all features and labels
            features = torch.cat(features_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            decoded, outputs = model(features, task)
            labels = labels.reshape(-1)  # Reshape labels to match outputs
            labels = labels.cpu().detach().numpy()

            if task == "sedimentation_rates":
                outputs = outputs.cpu().detach().numpy().reshape(-1)  # Reshape outputs to match labels
            else:
                outputs = outputs.cpu().detach().numpy().reshape(-1)  # Reshape outputs to match labels
            #labels = labels.cpu().detach().numpy().reshape(-1)
            mae = mean_absolute_error(labels, outputs)
            mse = mean_squared_error(labels, outputs)
            rmse = np.sqrt(mean_squared_error(labels, outputs))


            # Calculate R2 score and correlation coefficient
            r2 = r2_score(labels, outputs)

            correlation = np.corrcoef(labels, outputs)[0, 1]

            # Create scatter plot of labels vs outputs
            plt.figure(figsize=(8, 6))
            plt.scatter(labels, outputs, s=10)
            plt.xlabel("Labels")
            plt.ylabel("Outputs")
            plt.title(f"Scatter Plot - {task}")
            plt.text(0.5, 0.9, f"R2 Score: {r2:.4f}", ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.8, f"Correlation: {correlation:.4f}", ha='center', va='center', transform=plt.gca().transAxes)
            plt.show()


        # %%

        # Evaluate both training and test results
        print("\n=== Training Set Results ===")
        for task in train_dataloaders:
            mae, mse, corr, r2 = evaluate_model(model, train_dataloaders[task], task, device)
            print(f"\nTask: {task}")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"Correlation Coefficient: {corr:.4f}")
            print(f"R² Score: {r2:.4f}")

        print("\n=== Test Set Results ===")
        for task in test_dataloaders:
            mae, mse, corr, r2 = evaluate_model(model, test_dataloaders[task], task, device)
            print(f"\nTask: {task}")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"Correlation Coefficient: {corr:.4f}")
            print(f"R² Score: {r2:.4f}")
          
        # %%
        import torch
        from pathlib import Path
        import pickle
        import numpy as np
        from matplotlib import pyplot as plt
        #from sklearn.model_selection import StratifiedKFold, KFold
        #from sklearn.model_selection import train_test_split
        import os
        import gc

        #"""


        #os.chdir("/gxfs_work1/cau/sunms498/work/david/NN4SedRatePred/models/MLP/SedRate_MAR/models_torch/")


        from os import walk  
        feature_path = Path(Path() / "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men/")
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

        #model = FEA(layer_width)
        #model.load_state_dict(torch.load("/gxfs_work1/cau/sunms498/work/toc-neural-nets/SelectFeaturesRun/FEAmodel_test0_75.pth"))

        model.eval()
        #enable_dropout(model) #for mc dropout

        chunk_shape = [6,4320] #360 chunks

        np.random.seed(42)
        torch.manual_seed(42)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #'cuda:0' if torch.cuda.is_available() else 
        #print(physical_devices)
        model.to(device)
        print(device)
        model.double()

        first_run = True
        ii = 0

        # %%
        task_ids = ["sedimentation_rates", "toc"]

        features = np.load(Path(dataset_path / files[0]))

        chunk_shape = [6,4320] #360 chunks


        first_run = True
        ii = 0

        for count, file in enumerate(files):
            
            features = np.load(Path(dataset_path / file))
            features = np.divide((features - X_mean), X_std)

            features = torch.tensor(features)
            features = features.to(device)


            decoded, predictions_sedrate = model(features, task_id = "sedimentation_rates")
            predictions_sedrate = predictions_sedrate.cpu().detach().numpy()
            predictions_sedrate = predictions_sedrate.reshape(chunk_shape,  order = 'C')
            
            
            decoded, predictions_toc = model(features, task_id = "toc")
            predictions_toc = predictions_toc.cpu().detach().numpy()
            predictions_toc = predictions_toc.reshape(chunk_shape,  order = 'C')
            
            if first_run:
                prediction_map_sedrate = predictions_sedrate
                prediction_map_toc = predictions_toc
                first_run = False
            else:
                prediction_map_sedrate = np.append(prediction_map_sedrate, predictions_sedrate, axis = 0)
                prediction_map_toc = np.append(prediction_map_toc, predictions_toc, axis = 0)
                
            ii+=1
            print("prediction " + str(ii) + " done!")
            
            del(features)
            gc.collect()
            
        # land_file = Path(Path().resolve().parents[2] / "Data" / "restrepoKNNsTOC" / "island_map.npy")
        # land_map = np.load(land_file)
        # land_map[np.isnan(land_map)] = 1
        #prediction_map[land_map] = np.nan
        #plt.imshow(prediction_map)
        # np.save("Sed_regresion_deep_RestrepoFeats_AllVS0.2_nolimit", prediction_map)
        #prediction_map = np.load("MAR_prediction_map(log(x+1))RestrepoFeats__AllVS0.2.npy")
        # from matplotlib.colors import LogNorm
        #plot_TOC = 3
        # plot_sed = 100
        # 


        # %%
        land_file = DATA_RAW / "island_map.npy"
        land_map = np.load(land_file)
        land_map[np.isnan(land_map)] = 1
        prediction_map_sedrate = np.rot90(np.rot90(np.fliplr(prediction_map_sedrate)))
        prediction_map_toc = np.rot90(np.rot90(np.fliplr(prediction_map_toc)))
        #land_map = np.rot90(np.rot90(np.fliplr(land_map)))
        plt.imshow(land_map)
        plt.figure()
        plt.imshow(prediction_map_sedrate)
        plt.figure()
        plt.imshow(prediction_map_toc)


        # %%
        from matplotlib.colors import LogNorm
        prediction_map_sedrate[land_map] = 0
        prediction_map_toc[land_map] = 0
        plt.figure(figsize=[25,15], dpi=200)
        plt.imshow(prediction_map_sedrate,vmax=50, cmap = "cubehelix")
        plt.colorbar()
        plt.figure(figsize=[25,15], dpi=200)
        plt.imshow(prediction_map_toc, norm = LogNorm(vmin = 1e-1, vmax = 5), cmap = "cubehelix")
        plt.colorbar()

        # %%
        from matplotlib.colors import LogNorm
        prediction_map_sedrate[land_map] = 0
        prediction_map_toc[land_map] = 0
        plt.figure(figsize=[25,15], dpi=200)
        plt.imshow(prediction_map_sedrate, norm = LogNorm(vmin = 1e0, vmax = 1e3), cmap = "cubehelix")
        plt.colorbar()
        plt.figure(figsize=[25,15], dpi=200)
        plt.imshow(prediction_map_toc, norm = LogNorm(vmin = 1e0, vmax = 5), cmap = "cubehelix")
        plt.colorbar()

        # %%
        
        # Save the numpy arrays
        np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / f"prediction_map_MAR_entire_MTL_n_{count_mtl}.npy", prediction_map_sedrate)
        np.save(DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization" / f"prediction_map_toc_entire_MTL_n_{count_mtl}.npy", prediction_map_toc)
        #"""