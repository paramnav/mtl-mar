"""
Deep Neural Network Training Module for Sedimentation Rate Prediction

This module provides functions for building, training, and evaluating
feedforward neural networks (DNNs) for predicting sedimentation rates.
The architecture uses an encoder-decoder structure, though only the encoder
and supervised head are typically used for training.
"""

import torch
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
from os import walk  
import random
import torch.nn.functional as F

# Global lists to track training metrics across epochs
training_accuracy = []
training_losses = []
training_mae = []
learning_rate = [] 

# Global lists to track evaluation metrics
evaluation_losses = []
evaluation_mae = []
true_labels = []
pred_labels = []

##########################################################################
# Model Architecture Functions
##########################################################################

"""
Architecture Overview:
We present an autoencoder structure for the DNN. The decoder part is frozen and 
not trained, but can be used for unsupervised learning in the future. Though we 
show the entire autoencoder architecture, we typically use only the encoder and 
the supervised part for training and predictions. The entire architecture provides 
flexibility for other use cases where the amount of data available is low and 
requires an unsupervised approach.
"""

def initialize_weights(m):
    """
    Initialize weights for linear layers using Kaiming normal initialization.
    
    Args:
        m: PyTorch module (typically a Linear layer)
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    
def buildDNN(layer_width, features):
    """
    Build a Feedforward Encoder Architecture (FEA) neural network.
    
    The architecture consists of:
    - Encoder: Stack of Linear -> ReLU -> Dropout layers
    - Decoder: Empty Sequential (not used in supervised training)
    - Supervised head: Single linear layer for regression output
    
    Args:
        layer_width (list): List of integers specifying the width of each hidden layer
        features (numpy.ndarray): Feature array used to determine input dimension
        
    Returns:
        model: PyTorch model with double precision
    """
    class FEA(nn.Module):
        """
        Feedforward Encoder Architecture for sedimentation rate prediction.
        Uses an encoder-decoder structure, but only encoder + supervised head are trained.
        """
        def __init__(self, layer_width):
            super(FEA, self).__init__()
            
            # Dropout probability (reduced to 0.2 to help with training)
            self.do_prob = 0.2
            
            # Build encoder: stack of Linear -> ReLU -> Dropout layers
            self.encoder = nn.Sequential()
            
            # First layer: input features -> first hidden layer
            # Note: No BatchNorm on first layer (helps with skewed data)
            self.encoder.add_module('encoder_linear_0', nn.Linear(features.shape[1], layer_width[0]))
            self.encoder.add_module('encoder_relu_0', nn.ReLU())
            self.encoder.add_module('encoder_dropout_0', nn.Dropout(self.do_prob))
            
            # Middle layers: progressively reduce dimensionality
            for i in range(1, len(layer_width)):
                self.encoder.add_module(f'encoder_linear_{i}', 
                                    nn.Linear(layer_width[i-1], layer_width[i]))
                self.encoder.add_module(f'encoder_relu_{i}', 
                                    nn.ReLU())
                self.encoder.add_module(f'encoder_dropout_{i}', 
                                    nn.Dropout(self.do_prob))
            
            # Decoder: Empty Sequential (not used in supervised training)
            # Can be uncommented for unsupervised/autoencoder training
            self.decoder = nn.Sequential()
            
            # Supervised head: maps encoded representation to single output (sedimentation rate)
            self.supervised = nn.Sequential(
                nn.Linear(layer_width[-1], 1),
            )
            
        def forward(self, x):
            """
            Forward pass through the network.
            
            Args:
                x: Input features tensor
                
            Returns:
                targets: Predicted sedimentation rates
            """
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)  # Not used in supervised training
            targets = self.supervised(encoded)
            return targets 

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


def buildAttentionDNN(
    features,
    n_value_features=None,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ff_mult=2,
    dropout=0.2,
):
    """Feature-token self-attention regressor for value+mask MAR inputs.

    Drop-in for :func:`buildDNN`: flat ``(batch, n_features)`` → ``(batch, 1)``.
    When ``n_features == 2 * n_value_features``, each token is ``[value, mask]``.

    Paper FFNN+attention defaults: ``d_model=256, n_heads=8, n_layers=3,
    ff_mult=2, dropout=0.2`` on 226-d (113 values + 113 masks).
    """
    total_features = int(features.shape[1])
    if n_value_features is None:
        n_value_features = total_features // 2 if total_features % 2 == 0 else total_features
    has_mask = total_features == 2 * n_value_features
    in_channels = 2 if has_mask else 1

    class FeatureTokenTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_value_features = n_value_features
            self.has_mask = has_mask
            self.token_embed = nn.Linear(in_channels, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, n_value_features, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=max(ff_mult * d_model, d_model),
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.pool_norm = nn.LayerNorm(d_model)
            self.supervised = nn.Sequential(
                nn.Linear(d_model, max(16, d_model // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, d_model // 2), 1),
            )

        def forward(self, x):
            if self.has_mask:
                values = x[:, : self.n_value_features]
                masks = x[:, self.n_value_features :]
                tokens = torch.stack([values, masks], dim=-1)
            else:
                tokens = x[:, : self.n_value_features].unsqueeze(-1)
            embedded = self.token_embed(tokens) + self.pos_embed
            encoded = self.encoder(embedded)
            pooled = self.pool_norm(encoded.mean(dim=1))
            return self.supervised(pooled)

    model = FeatureTokenTransformer()
    model = model.double()
    initialize_weights(model.token_embed)
    for layer in model.supervised:
        initialize_weights(layer)
    return model

# Lists to track training metrics
training_accuracy = []
training_losses = []
training_mae = []
learning_rate = []

##########################################################################
# Training Functions
##########################################################################

def train(model, epoch, trainloader, train_step, change_lr, count, optimizer, lr_scheduler, loss_fn):
    """
    Train the model for one epoch.
    
    Training steps:
    - train_step == 1: Autoencoder reconstruction (decoder training)
    - train_step == 2: Supervised learning (encoder + supervised head)
    - train_step == 3: Combined reconstruction + supervised learning
    
    Args:
        model: PyTorch model to train
        epoch (int): Current epoch number
        trainloader (DataLoader): Training data loader
        train_step (int): Current training step (1-3)
        change_lr (bool): Whether to update learning rate scheduler
        count (int): Current count for progress tracking
        optimizer: PyTorch optimizer
        lr_scheduler: Learning rate scheduler
        loss_fn: Loss function
    """
    model.train()  # Set model to training mode
    running_loss = 0
    counter = 0 
    mae_epoch = 0

    # Iterate over training batches
    for features, labels in trainloader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            
        # Convert to double precision
        features = features.to(dtype=torch.double)
        labels = labels.to(dtype=torch.double)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        targets = model(features)

        # Calculate loss based on training step
        # Note: train_step 1 and 3 reference 'decoded' which may not be defined
        # depending on model architecture - this is preserved as-is for compatibility
        if train_step == 1:
            # Step 1: Autoencoder reconstruction loss
            loss = loss_fn(features, decoded)
            mae_epoch += torch.mean(torch.abs(decoded - features))
            
        elif train_step == 2:
            # Step 2: Supervised learning (most commonly used)
            labels = torch.reshape(labels, (labels.shape[0], 1))
            targets = torch.reshape(targets, (targets.shape[0], 1))
            loss = loss_fn(targets, labels)
            mae_epoch += torch.mean(torch.abs(targets - labels))
            
        elif train_step == 3:
            # Step 3: Combined reconstruction + supervised learning
            labels = torch.reshape(labels, (labels.shape[0], 1))
            targets = torch.reshape(targets, (targets.shape[0], 1))
            loss = loss_fn(targets, labels) + loss_fn(features, decoded)
            mae_epoch += torch.mean(torch.abs(targets - labels)) + torch.mean(torch.abs(decoded - features))

        # Backward pass: compute gradients
        loss.backward()
        optimizer.step()
        
        # Update learning rate if requested
        if change_lr:
            lr_scheduler.step(loss)
        
        # Accumulate metrics
        running_loss += loss.item()
        counter += 1

    # Calculate epoch-level metrics
    training_loss = running_loss / len(trainloader)
    mae = mae_epoch / counter
    lr = optimizer.param_groups[0]['lr']
    
    # Store metrics in global lists
    learning_rate.append(lr)
    training_mae.append(mae.cpu().detach().numpy())
    training_losses.append(training_loss)

    # Print progress
    print(f'epoch: {epoch} | Training Loss: {training_loss:.3f} | MAE: {mae:.3f} | LR: {lr:.7f} | count: {count}/36 |')

def validation(model, epoch, testloader, train_step, count, loss_fn):
    """
    Run validation on the model.
    
    Evaluates the model on validation/test data without updating weights.
    Uses the same loss calculation as training based on train_step.
    
    Args:
        model: PyTorch model to evaluate
        epoch (int): Current epoch number
        testloader (DataLoader): DataLoader for validation data
        train_step (int): Current training step (1, 2, or 3) - determines loss calculation
        count (int): Counter for logging
        loss_fn: Loss function
        
    Returns:
        testing_loss (float): Average validation loss for the epoch
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    running_loss = 0
    counter = 0
    mae_epoch = 0

    # Disable gradient computation for validation (faster, uses less memory)
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
            
            # Calculate loss based on training step (same logic as training)
            # Note: train_step 1 and 3 reference 'decoded' which may not be defined
            if train_step == 1:
                # Step 1: Autoencoder reconstruction loss
                loss = loss_fn(features, decoded)
                mae_epoch += torch.mean(torch.abs(decoded - features))
                
            elif train_step == 2:
                # Step 2: Supervised learning loss
                labels = torch.reshape(labels, (labels.shape[0], 1))
                targets = torch.reshape(targets, (targets.shape[0], 1))
                loss = loss_fn(targets, labels)
                mae_epoch += torch.mean(torch.abs(targets - labels))
                
            elif train_step == 3:
                # Step 3: Combined reconstruction + supervised learning loss
                labels = torch.reshape(labels, (labels.shape[0], 1))
                targets = torch.reshape(targets, (targets.shape[0], 1))
                loss = loss_fn(targets, labels) + loss_fn(features, decoded)
                mae_epoch += torch.mean(torch.abs(targets - labels)) + torch.mean(torch.abs(decoded - features))
            
            # Accumulate metrics
            running_loss += loss.item()
            counter += 1

    # Calculate validation metrics
    testing_loss = running_loss / len(testloader)
    mae = mae_epoch / counter
    
    # Store metrics in global lists
    evaluation_losses.append(testing_loss)
    evaluation_mae.append(mae)
    
    # Print validation results
    print(f'epoch: {epoch} | Testing Loss: {testing_loss:.3f} | MAE: {mae:.3f} | count: {count}/36 |')
    return testing_loss

##########################################################################
# Data Preprocessing Functions
##########################################################################

def preprocess_features_labels(features, labels, lats, lons, X_mean, X_std):
    """
    Preprocesses features and labels for training.
    
    This function performs several preprocessing steps:
    1. Removes rows and columns with NaN values
    2. Normalizes features using provided mean and std
    3. Handles duplicate feature rows by either removing them or averaging labels
    
    Parameters:
        features (numpy.ndarray): Input features (n_samples, n_features)
        labels (numpy.ndarray): Input labels (n_samples,)
        lats (numpy.ndarray): Latitude coordinates (n_samples,)
        lons (numpy.ndarray): Longitude coordinates (n_samples,)
        X_mean (numpy.ndarray): Mean values for feature normalization (n_features,)
        X_std (numpy.ndarray): Standard deviation values for normalization (n_features,)

    Returns:
        features (numpy.ndarray): Preprocessed features
        labels (numpy.ndarray): Preprocessed labels
        lats (numpy.ndarray): Filtered latitude coordinates
        lons (numpy.ndarray): Filtered longitude coordinates
    """
    # Step 1: Remove rows (samples) with NaN values in any feature
    nan_rows = np.isnan(features).any(axis=1)
    np.save("FeatureNanRows", nan_rows)  # Save mask for reference
    features = features[~nan_rows, :]
    labels = labels[~nan_rows]
    lats = lats[~nan_rows]
    lons = lons[~nan_rows]

    # Step 2: Remove columns (features) with NaN values
    nan_rows = np.isnan(features).any(axis=0)
    np.save("FeatureNanRows", nan_rows)  # Save mask for reference
    features = features[:, ~nan_rows]

    # Step 3: Normalize features using z-score normalization
    features = np.divide((features - X_mean), X_std)

    # Step 4: Handle duplicate feature rows
    # If duplicate rows have similar labels, average them
    # If duplicate rows have very different labels, remove them (likely data errors)
    arr = features
    unique_rows, row_counts = np.unique(arr, axis=0, return_counts=True)
    duplicate_indices = np.where(row_counts > 1)[0]
    groups = np.split(unique_rows, np.where(np.diff(unique_rows, axis=0).any(axis=1))[0] + 1)

    for idx in duplicate_indices:
        group_idx = [i for i, group in enumerate(groups) if (group == unique_rows[idx]).all()][0]
        group_labels = labels[np.where((arr == unique_rows[idx]).all(axis=1))[0]]
        
        # If labels vary significantly (>20% of max), remove all duplicates (likely errors)
        if np.std(group_labels) > 0.2 * np.max(group_labels):
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx, :] = np.nan
            labels[row_idx] = np.nan
        else:
            # If labels are similar, keep first occurrence and average labels
            row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
            arr[row_idx[1:], :] = np.nan  # Remove duplicate rows except the first
            
            # Average labels for the first occurrence, set others to NaN
            for z in range(len(row_idx)):
                if z == 0:
                    labels[row_idx[z]] = np.mean(group_labels)  # Average labels
                else:
                    labels[row_idx[z]] = np.nan

    # Step 5: Remove all rows that now have NaN values (after duplicate handling)
    num_nan_values_before = np.sum(np.isnan(arr), axis=1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    num_nan_values_after = np.sum(np.isnan(arr), axis=1)
    
    # Filter coordinates to match remaining samples
    lats = lats[~np.isnan(labels)]
    lons = lons[~np.isnan(labels)]
    labels = labels[~np.isnan(labels)]

    features = arr

    # Print summary of preprocessing
    print(f"Number of rows in features before: {len(num_nan_values_before)}")
    print(f"Number of rows in features after: {len(num_nan_values_after)}")
    print(f"Number of rows in labels: {len(labels)}")
    print(f"Number of rows in lats: {len(lats)}")
    print(f"Number of rows in lons: {len(lons)}")

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

def train_model(
    model,
    loss_fn,
    trainloader,
    testloader,
    num_epochs=500,
    lr=0.01,
    weight_decay=0.0,
):
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
    lr (float): Adam learning rate (default 0.01 for legacy FFNN notebooks;
        paper attention / nn4sedratetoc use 1e-4).
    weight_decay (float): Adam L2 penalty (paper attention uses 1e-4).
    """
    train_step = 2
    # Freeze the parameters of the decoder
    #for param in model.decoder.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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