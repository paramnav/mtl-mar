
import torch
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.model_selection import StratifiedKFold, KFold
#from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
from os import walk  
import random
import torch.nn.functional as F

training_accuracy = []
training_losses = []
training_mae = []
learning_rate =  [] 

evaluation_losses = []
evaluation_mae = []
true_labels = []
pred_labels = []

##########################################################################
#build_dnn()
"""
We present here an autoencoder structure here for the DNN.
The decoder part is frozen and is not trained, but can be used for unsupervised learning for future.
than the supervised approach. Though here, we show the entire autoencoder architecture, we would be using only the encoder and the supervised part for training and predictions. But the entire architecture provides flexibility for other use cases where the amount of data available is low and requires unsupervised approach.
"""

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    
def buildDNN(layer_width, features):
    class FEA(nn.Module):
        def __init__(self, layer_width):
            super(FEA, self).__init__()

            
            # Reduce dropout probability
            self.do_prob = 0.2  # or even 0.05
            
            self.encoder = nn.Sequential()
            
            # First layer without BatchNorm (helps with skewed data)
            self.encoder.add_module('encoder_linear_0', nn.Linear(features.shape[1], layer_width[0]))
            self.encoder.add_module('encoder_relu_0', nn.ReLU())
            self.encoder.add_module('encoder_dropout_0', nn.Dropout(self.do_prob))
            
            # Middle layers
            for i in range(1, len(layer_width)):
                self.encoder.add_module(f'encoder_linear_{i}', 
                                    nn.Linear(layer_width[i-1], layer_width[i]))
                self.encoder.add_module(f'encoder_relu_{i}', 
                                    nn.ReLU())
                self.encoder.add_module(f'encoder_dropout_{i}', 
                                    nn.Dropout(self.do_prob))
            self.decoder = nn.Sequential()

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
                #nn.ReLU()
            )
            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            targets = self.supervised(encoded)
            return targets#decoded, 

    def initialize_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0)

            

    input_dim = features.shape[1]  # Replace with the actual input dimension






    #for count, layer_width in enumerate(layer_width_stack):
    # Model Initialization
    model = FEA(layer_width)
    model = model.double()
    model.apply(initialize_weights)

    return model

# Lists to track training metrics
training_accuracy = []
training_losses = []
training_mae = []
learning_rate = []

def train(model, epoch, trainloader, train_step, change_lr, count, optimizer, lr_scheduler, loss_fn):
    """
    Train the model for one epoch.
    
    Args:
        epoch (int): Current epoch number
        trainloader (DataLoader): Training data loader
        train_step (int): Current training step (1-3)
        change_lr (bool): Whether to update learning rate
        count (int): Current count for progress tracking
    """
    model.train()
    running_loss = 0
    counter = 0 
    mae_epoch = 0

    for features, labels in trainloader:
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            
        features = features.to(dtype=torch.double)
        labels = labels.to(dtype=torch.double)

        optimizer.zero_grad()
        
        targets = model(features)

        # Calculate loss based on training step
        if train_step == 1:
            loss = loss_fn(features, decoded)
            mae_epoch += torch.mean(torch.abs(decoded - features))
            
        elif train_step == 2:
            labels = torch.reshape(labels, (labels.shape[0], 1))
            targets = torch.reshape(targets, (targets.shape[0], 1))
            loss = loss_fn(targets, labels)
            mae_epoch += torch.mean(torch.abs(targets - labels))
            
        elif train_step == 3:
            labels = torch.reshape(labels, (labels.shape[0], 1))
            targets = torch.reshape(targets, (targets.shape[0], 1))
            loss = loss_fn(targets, labels) + loss_fn(features, decoded)
            mae_epoch += torch.mean(torch.abs(targets - labels)) + torch.mean(torch.abs(decoded - features))

        loss.backward()
        optimizer.step()
        
        if change_lr:
            lr_scheduler.step(loss)
        
        running_loss += loss.item()
        counter += 1

    # Calculate epoch metrics
    training_loss = running_loss / len(trainloader)
    mae = mae_epoch / counter
    lr = optimizer.param_groups[0]['lr']
    
    # Store metrics
    learning_rate.append(lr)
    training_mae.append(mae.cpu().detach().numpy())
    training_losses.append(training_loss)

    print(f'epoch: {epoch} | Training Loss: {training_loss:.3f} | MAE: {mae:.3f} | LR: {lr:.7f} | count: {count}/36 |')

def validation(model, epoch, testloader, train_step, count, loss_fn):
    """Run validation on the model.
    
    Args:
        epoch (int): Current epoch number
        testloader (DataLoader): DataLoader for validation data
        train_step (int): Current training step (1, 2, or 3)
        count (int): Counter for logging
        store_labels (bool): Whether to store true/predicted labels
    """
    model.eval()
    running_loss = 0
    counter = 0
    mae_epoch = 0

    with torch.no_grad():
        for features, labels in testloader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            # Convert to double precision
            features = features.to(dtype=torch.double)
            labels = labels.to(dtype=torch.double)
            
            # Get model predictions
            targets = model(features)
            
            # Calculate loss based on training step
            if train_step == 1:
                loss = loss_fn(features, decoded)
                mae_epoch += torch.mean(torch.abs(decoded - features))
                
            elif train_step == 2:
                labels = torch.reshape(labels, (labels.shape[0], 1))
                targets = torch.reshape(targets, (targets.shape[0], 1))
                loss = loss_fn(targets, labels)
                mae_epoch += torch.mean(torch.abs(targets - labels))
                
            elif train_step == 3:
                labels = torch.reshape(labels, (labels.shape[0], 1))
                targets = torch.reshape(targets, (targets.shape[0], 1))
                loss = loss_fn(targets, labels) + loss_fn(features, decoded)
                mae_epoch += torch.mean(torch.abs(targets - labels)) + torch.mean(torch.abs(decoded - features))
            
            running_loss += loss.item()
            counter += 1

    # Calculate validation metrics
    testing_loss = running_loss / len(testloader)
    mae = mae_epoch / counter
    
    # Store metrics
    evaluation_losses.append(testing_loss)
    evaluation_mae.append(mae)


    
    print(f'epoch: {epoch} | Testing Loss: {testing_loss:.3f} | MAE: {mae:.3f} | count: {count}/36 |')
    return testing_loss

########################################################################

def preprocess_features_labels(features, labels, lats, lons, X_mean, X_std):
    """
    Preprocesses features and labels.

    Parameters:
    features (numpy.ndarray): Input features.
    labels (numpy.ndarray): Input labels.
    X_mean (numpy.ndarray): Mean values for normalization.
    X_std (numpy.ndarray): Standard deviation values for normalization.

    Returns:
    torch Tensor: Preprocessed features.
    torch Tensor: Preprocessed labels.
    """

    # Save rows with NaN values
    nan_rows = np.isnan(features).any(axis=1)
    np.save("FeatureNanRows", nan_rows)
    features = features[~nan_rows, :]
    labels = labels[~nan_rows]
    lats = lats[~nan_rows]
    lons = lons[~nan_rows]
    

    # Drop features with NaN values
    nan_rows = np.isnan(features).any(axis=0)
    np.save("FeatureNanRows", nan_rows)
    features = features[:, ~nan_rows]

    # Normalize features
    features = np.divide((features - X_mean), X_std)

    # Replace duplicate rows with NaN values and adjust labels
    arr = features
    unique_rows, row_counts = np.unique(arr, axis=0, return_counts=True)
    duplicate_indices = np.where(row_counts > 1)[0]
    groups = np.split(unique_rows, np.where(np.diff(unique_rows, axis=0).any(axis=1))[0] + 1)

    for idx in duplicate_indices:
        group_idx = [i for i, group in enumerate(groups) if (group == unique_rows[idx]).all()][0]
        group_labels = labels[np.where((arr == unique_rows[idx]).all(axis=1))[0]]
        # print(group_labels)
        # print(group_idx)
        # print(unique_rows[idx])
        if np.std(group_labels) > 0.2 * np.max(group_labels):
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx,:] = np.nan
            labels[row_idx] = np.nan
            #print(f"Duplicate row with index {row_idx} was in group {group_idx} and was replaced with NaN values")

        else:
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx[1:], :] = np.nan  # remove duplicate rows except the first occurrence

            for z in range(len(row_idx)):
                if z == 0:
                    labels[row_idx[z]] = np.mean(group_labels)  # replace the label of the first occu
                else:
                    labels[row_idx[z]] = np.nan

            # print(row_idx)

    # remove rows with NaN values
    num_nan_values_before = np.sum(np.isnan(arr), axis=1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    num_nan_values_after = np.sum(np.isnan(arr), axis=1)
    print(lats.shape)
    print(labels.shape)
    lats = lats[~np.isnan(labels)]
    lons = lons[~np.isnan(labels)]
    labels = labels[~np.isnan(labels)]

    features = arr

    print(f"Number of rows in features before: {len(num_nan_values_before)}")
    print(f"Number of rows in features after: {len(num_nan_values_after)}")
    print(f"Number of rows in labels: {len(labels)}")
    print(f"Number of rows in lats: {len(lats)}")
    print(f"Number of rows in lons: {len(lons)}")

    #features = torch.tensor(features)
    #labels = torch.tensor(labels)

    return features, labels, lats, lons

def split_train_test_data(features, labels, test_size_ratio=0.25, random_seed=42, stratify=None):
    """
    Splits features and labels into train and test numpy arrays.

    Parameters:
    features (numpy.ndarray): Input features.
    labels (numpy.ndarray): Input labels.
    test_size_ratio (float): Ratio of test set size to total dataset size (default: 0.25).
    random_seed (int): Random seed for reproducibility.
    stratify (array-like, optional): If not None, data is split in a stratified fashion, using this as class labels.

    Returns:
    train_features, train_labels, test_features, test_labels (numpy.ndarray): Training and test arrays
    """
    num_samples = features.shape[0]
    testsize = int(num_samples * test_size_ratio)
    trainsize = num_samples - testsize

    # Optionally stratify
    if stratify is not None:
        # Use numpy for stratified split
        from sklearn.model_selection import train_test_split
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=test_size_ratio, random_state=random_seed, stratify=stratify
        )
    else:
        # Shuffle indices
        np.random.seed(random_seed)
        perm = np.random.permutation(num_samples)
        train_idx = perm[:trainsize]
        test_idx = perm[trainsize:]
        
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

    return train_features, train_labels, test_features, test_labels

def create_dataloaders(train_features, train_labels, test_features, test_labels, batch_size=200, shuffle_train=True, shuffle_test=False):
    """
    Creates DataLoaders from training and test arrays.

    Parameters:
    train_features (numpy.ndarray): Training features
    train_labels (numpy.ndarray): Training labels  
    test_features (numpy.ndarray): Test features
    test_labels (numpy.ndarray): Test labels
    batch_size (int): Batch size for DataLoader.
    shuffle_train (bool): Whether to shuffle the training dataset.
    shuffle_test (bool): Whether to shuffle the test dataset.

    Returns:
    trainloader (DataLoader): DataLoader for training data.
    testloader (DataLoader): DataLoader for test data.
    """
    # Convert to tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    if torch.cuda.is_available():
        train_features = train_features.cuda()
        train_labels = train_labels.cuda()
        test_features = test_features.cuda()
        test_labels = test_labels.cuda()

    # Create datasets
    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)

    # Create dataloaders
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test)

    # Print summary
    print("Train data size")
    print(len(train_set))
    print("Test data size")
    print(len(test_set))

    return trainloader, testloader

def create_train_test_loader(features, labels, batch_size = 200, test_size_ratio=1/7, shuffle=True):
    """
    Creates train and test loaders.

    Parameters:
    features (numpy.ndarray): Input features.
    labels (numpy.ndarray): Input labels.
    batch_size (int): Batch size for DataLoader.
    test_size_ratio (float): Ratio of test set size to total dataset size (default: 0.2).
    shuffle (bool): Whether to shuffle the dataset before creating loaders (default: True).

    Returns:
    torch.utils.data.DataLoader: Train loader.
    torch.utils.data.DataLoader: Test loader.
    """

    dataset=TensorDataset(features, labels)
    dataloader=DataLoader(dataset, batch_size=batch_size)



    testsize=int(features.shape[0]*test_size_ratio)
    trainsize=features.shape[0]-testsize

  
    train_set, test_set = torch.utils.data.random_split(dataset, [trainsize, testsize], generator=torch.Generator().manual_seed(42))



    trainloader= torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader= torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    num_channels = features.shape[1]

    #####################################################summary of training set
    dataiter = iter(trainloader)
    features, labels = next(dataiter)
    
    print("Train data size")
    print(trainsize)
    print("Test data size")
    print(testsize)
    ######################################################


    return trainloader, testloader

def train_model(model, loss_fn, trainloader, testloader, num_epochs=500):
    """
    Train the model.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    loss_fn: Loss function used for training.
    optimizer: Optimizer used for training.
    train_loader: DataLoader for training data.
    test_loader: DataLoader for test data.
    train_step (int): Training step (default: 2).
    num_epochs (int): Number of epochs for training (default: 500).
    """
    train_step = 2
    # Freeze the parameters of the decoder
    #for param in model.decoder.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, 
                                                          patience=50, threshold=0.0001, threshold_mode='rel', 
                                                          cooldown=0, min_lr=0.00001, eps=1e-08)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # Training loop
    for epoch in range(num_epochs):
        change_LR = True
        count = 1
        # Train for one epoch
        train(model, epoch, trainloader, train_step, change_LR, count, optimizer, lr_scheduler, loss_fn)
        val_loss = validation(model, epoch, testloader, train_step, count, loss_fn)

        # Early stopping logic
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} to avoid overfitting.")
                break

    return training_accuracy, training_losses, training_mae, learning_rate, evaluation_losses, evaluation_mae, true_labels, pred_labels

def evaluate_model_performance(model, testloader):
    """
    Evaluate model performance on test data
    """
    model.eval()
    
    test_targets = []
    test_labels = []
    
    with torch.no_grad():
        for features, labels in testloader:
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            features = features.to(dtype=torch.double)
            labels = labels.to(dtype=torch.double)
            
            # Get predictions
            targets = model(features)
            
            test_targets.append(targets)
            test_labels.append(labels)
    
    # Convert to numpy arrays
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy().flatten()
    test_targets = torch.cat(test_targets, dim=0).detach().cpu().numpy().flatten()
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    corr_coef = np.ma.corrcoef(test_labels, test_targets)[0, 1]
    mae = mean_absolute_error(test_labels, test_targets)
    mse = mean_squared_error(test_labels, test_targets)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"Test Set Performance:")
    print(f"Correlation coefficient: {corr_coef:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_labels, test_targets, alpha=0.6)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Values")
    plt.title("Model Performance on Test Set")
    
    # Add diagonal line for reference
    min_val = min(test_labels.min(), test_targets.min())
    max_val = max(test_labels.max(), test_targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'correlation': corr_coef,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'test_labels': test_labels,
        'test_targets': test_targets
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass