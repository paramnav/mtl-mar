
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


density_fileidx = 70
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

"""
def buildDNN(layer_width, features, num_shared_layers):
    class MTLNN(nn.Module):
        def __init__(self, features, layer_width, num_shared_layers=num_shared_layers):
            super(MTLNN, self).__init__()

            self.do_prob = 0.2

            # If all layers are in the encoder and only one task, match FEA structure
            if num_shared_layers == len(layer_width):
                # Encoder: stack of Linear->ReLU->Dropout layers
                self.encoder = nn.Sequential()
                self.encoder.add_module('encoder_linear_0', nn.Linear(features.shape[1], layer_width[0]))
                self.encoder.add_module('encoder_relu_0', nn.ReLU())
                self.encoder.add_module('encoder_dropout_0', nn.Dropout(self.do_prob))
                for i in range(1, len(layer_width)):
                    self.encoder.add_module(f'encoder_linear_{i}', nn.Linear(layer_width[i-1], layer_width[i]))
                    self.encoder.add_module(f'encoder_relu_{i}', nn.ReLU())
                    self.encoder.add_module(f'encoder_dropout_{i}', nn.Dropout(self.do_prob))
                self.decoder = nn.Sequential()  # Not used for supervised
                # Output head: single Linear layer from last encoder size to 1
                self.task_sedimentation_rates_head = nn.Sequential(
                    nn.Linear(layer_width[-1], 1)
                )
                # If you want to support a second task, you can add another head here
                self.task_toc_head = nn.Sequential(
                    nn.Linear(layer_width[-1], 1)
                )
            else:
                # Original MTLNN structure for multi-task with split encoder/head
                self.encoder = self._make_layers(features.shape[1], layer_width[:num_shared_layers])
                self.decoder = self._make_layers(layer_width[num_shared_layers-1], layer_width[:num_shared_layers][::-1], final_output_size=features.shape[1])
                self.task_sedimentation_rates_head = self._make_layers(layer_width[num_shared_layers-1], layer_width[num_shared_layers:], final_output_size=1)
                self.task_toc_head = self._make_layers(layer_width[num_shared_layers-1], layer_width[num_shared_layers:], final_output_size=1)

        def _make_layers(self, input_size, layer_width, final_output_size=None):
            layers = []
            for i in range(len(layer_width)):
                layers.append(nn.Linear(input_size, layer_width[i]))
                #layers.append(nn.BatchNorm1d(layer_width[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.do_prob))
                input_size = layer_width[i]
            if final_output_size is not None:
                layers.append(nn.Linear(input_size, final_output_size))
            return nn.Sequential(*layers)

        def forward(self, x, task_id):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)  # decoder not used for supervised
            if task_id == "sedimentation_rates":
                task_output = self.task_sedimentation_rates_head(encoded)
            elif task_id == "toc":
                task_output = self.task_toc_head(encoded)
            elif task_id == "reconstruction":
                task_output = self.task_sedimentation_rates_head(encoded)
            else:
                raise ValueError("Invalid task_id. Use sedimentation_rates or toc.")
            return decoded, task_output
        # Assuming features is already defined
    input_dim = features.shape[1]  # Replace with the actual input dimension
    model = MTLNN(features, layer_width, num_shared_layers)
    model.apply(initialize_weights)
    model.double()

    return model
"""
#without the reconstruction/decoder 
def buildDNN(layer_width, features, num_shared_layers):
    class MTLNN(nn.Module):
        def __init__(self, features, layer_width, num_shared_layers=num_shared_layers):
            super(MTLNN, self).__init__()

            self.do_prob = 0.2

            # --- BRANCH 1: All layers shared (Single Task / FEA structure) ---
            if num_shared_layers == len(layer_width):
                self.encoder = nn.Sequential()
                self.encoder.add_module('encoder_linear_0', nn.Linear(features.shape[1], layer_width[0]))
                self.encoder.add_module('encoder_relu_0', nn.ReLU())
                self.encoder.add_module('encoder_dropout_0', nn.Dropout(self.do_prob))
                
                for i in range(1, len(layer_width)):
                    self.encoder.add_module(f'encoder_linear_{i}', nn.Linear(layer_width[i-1], layer_width[i]))
                    self.encoder.add_module(f'encoder_relu_{i}', nn.ReLU())
                    self.encoder.add_module(f'encoder_dropout_{i}', nn.Dropout(self.do_prob))
                
                # Heads take the last layer output directly
                self.task_sedimentation_rates_head = nn.Sequential(
                    nn.Linear(layer_width[-1], 1)
                )
                self.task_toc_head = nn.Sequential(
                    nn.Linear(layer_width[-1], 1)
                )

            # --- BRANCH 2: Shared Layers + Task Specific Layers ---
            else:
                # 1. Create Shared Encoder
                self.encoder = self._make_layers(features.shape[1], layer_width[:num_shared_layers])
                
                # 2. Create Task Heads (Encoder Output -> Remaining Layers -> 1)
                # Note: We removed self.decoder here
                self.task_sedimentation_rates_head = self._make_layers(
                    layer_width[num_shared_layers-1], 
                    layer_width[num_shared_layers:], 
                    final_output_size=1
                )
                self.task_toc_head = self._make_layers(
                    layer_width[num_shared_layers-1], 
                    layer_width[num_shared_layers:], 
                    final_output_size=1
                )

        def _make_layers(self, input_size, layer_width, final_output_size=None):
            layers = []
            for i in range(len(layer_width)):
                layers.append(nn.Linear(input_size, layer_width[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.do_prob))
                input_size = layer_width[i]
            if final_output_size is not None:
                layers.append(nn.Linear(input_size, final_output_size))
            return nn.Sequential(*layers)

        def forward(self, x, task_id):
            # 1. Encode
            encoded = self.encoder(x)
            
            # 2. Decode step is REMOVED
            
            # 3. Task Specific Head
            if task_id == "sedimentation_rates":
                task_output = self.task_sedimentation_rates_head(encoded)
            elif task_id == "toc":
                task_output = self.task_toc_head(encoded)
            else:
                raise ValueError("Invalid task_id. Use sedimentation_rates or toc.")
            decoded = 0
            # Return only the task prediction (removed 'decoded')
            return decoded, task_output

    model = MTLNN(features, layer_width, num_shared_layers)
    # model.apply(initialize_weights) # Uncomment if you have this function
    model.double()

    return model


def buildAttentionMTL(
    features,
    n_value_features=None,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ff_mult=2,
    dropout=0.2,
):
    """Feature-token attention MTL (shared encoder + MAR/TOC heads).

    Same ``forward(x, task_id) -> (decoded, task_output)`` contract as
    :func:`buildDNN`. Paper MTL+attention defaults: ``d_model=128, n_heads=8,
    n_layers=2, ff_mult=2, dropout=0.2`` on 226-d value+mask inputs.
    """
    total_features = int(features.shape[1])
    if n_value_features is None:
        n_value_features = total_features // 2 if total_features % 2 == 0 else total_features
    has_mask = total_features == 2 * n_value_features
    in_channels = 2 if has_mask else 1
    head_hidden = max(16, d_model // 2)

    class FeatureTokenAttentionMTL(nn.Module):
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
            self.task_sedimentation_rates_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            self.task_toc_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        def _encode(self, x):
            if self.has_mask:
                values = x[:, : self.n_value_features]
                masks = x[:, self.n_value_features :]
                tokens = torch.stack([values, masks], dim=-1)
            else:
                tokens = x[:, : self.n_value_features].unsqueeze(-1)
            embedded = self.token_embed(tokens) + self.pos_embed
            encoded = self.encoder(embedded)
            return self.pool_norm(encoded.mean(dim=1))

        def forward(self, x, task_id):
            pooled = self._encode(x)
            if task_id == "sedimentation_rates":
                task_output = self.task_sedimentation_rates_head(pooled)
            elif task_id == "toc":
                task_output = self.task_toc_head(pooled)
            else:
                raise ValueError("Invalid task_id. Use sedimentation_rates or toc.")
            return 0, task_output

    model = FeatureTokenAttentionMTL()
    model = model.double()
    initialize_weights(model.token_embed)
    for head in (model.task_sedimentation_rates_head, model.task_toc_head):
        for layer in head:
            initialize_weights(layer)
    return model


def buildDNN2(layer_width, features, num_task_specific_layers):
    class FFNN(nn.Module):
        """Simple FF network with multiple outputs.
        """
        def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs = 1,
            dropout_rate=0.1,
        ):
            """

            :param input_size: input size
            :param hidden_size: common hidden size for all layers
            :param n_hidden: number of hidden layers
            :param n_outputs: number of outputs
            :param dropout_rate: dropout rate
            """
            super().__init__()
            #assert 0 <= dropout_rate < 1
            self.input_size = input_size

            h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]
            

            

            self.hidden = nn.ModuleList()
            for k in range(len(h_sizes) - 1):
                self.hidden.append(
                    nn.Linear(
                        h_sizes[k],
                        h_sizes[k + 1]
                    )
                )

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout_rate)
            self.double()


        def forward(self, x):

            for layer in self.hidden[:-1]:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)

            return self.hidden[-1](x)
        


    class TaskIndependentLayers(nn.Module):
        """NN for MTL with hard parameter sharing
        """

        def __init__(
                self,
                input_size,
                hidden_size,
                n_hidden,
                n_outputs,
                dropout_rate=0.1,
        ):

            super().__init__()
            self.n_outputs = n_outputs
            self.task_nets = nn.ModuleList()
            for _ in range(n_outputs):
                self.task_nets.append(
                    FFNN(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_hidden=n_hidden,
                        n_outputs=1,
                        dropout_rate=dropout_rate,
                    )
                )

    def forward(self, x):

        return torch.cat(
            tuple(task_model(x) for task_model in self.task_nets),
            dim=1
        )


    class HardSharing(nn.Module):
        """FFNN with hard parameter sharing
        """

        def __init__(
            self,
            hidden_size,
            n_hidden,
            n_outputs,
            n_task_specific_layers=[0,0],
            task_specific_hidden_size=None,
            dropout_rate=0.1,
        ):

            super().__init__()
            if task_specific_hidden_size is None:
                task_specific_hidden_size = hidden_size

            self.model = nn.Sequential()

            self.model.add_module(
                'hard_sharing',
                FFNN(
                    input_size=features.shape[1],
                    hidden_size=hidden_size,
                    n_hidden=n_hidden,
                    n_outputs=hidden_size,
                    dropout_rate=dropout_rate
                )
            )
            
            if n_task_specific_layers > 0:
                # if n_task_specific_layers == 0 than the task specific mapping is linear and
                # constructed as the product of last layer is the 'hard_sharing' and the linear layer
                # in 'task_specific', with no activation or dropout
                self.model.add_module('relu', nn.ReLU())
                self.model.add_module('dropout', nn.Dropout(p=dropout_rate))

            self.model.add_module(
                'task_specific',
                TaskIndependentLayers(
                    input_size=hidden_size,
                    hidden_size=task_specific_hidden_size,
                    n_hidden=n_task_specific_layers,
                    n_outputs=n_outputs,
                    dropout_rate=dropout_rate
                )
            )

        def forward(self, x):
            return self.model(x)
            
    model = HardSharing(
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        n_task_specific_layers=n_task_specific_layers_val,
        task_specific_hidden_size=task_specific_hidden_size,
        dropout_rate=dropout_rate,
        )
    model.apply(initialize_weights)
    model.double()
    return model


def train(model, epoch, trainloader, train_step, change_lr, count, optimizer, lr_scheduler, loss_fn):
    """
    Args:
        train_dataloaders: dict with keys 'sedimentation_rates' and 'toc', each a DataLoader
    """
    
    model.train()  # Set the model to training mode

    train_errors = {
        'sedimentation_rates': [],
        'toc': []
    }

    if train_step == 1:
        # ... (Step 1 logic remains the same) ...
        task = 'reconstruction'
        # (Keeping your existing step 1 code...)
        for param in model.task_toc_head.parameters(): param.requires_grad = True
        for param in model.task_sedimentation_rates_head.parameters(): param.requires_grad = True
        for features, labels in trainloader:
            if not isinstance(features, torch.Tensor): features = torch.tensor(features)
            if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)
            if torch.cuda.is_available(): features = features.cuda(); labels = labels.cuda()
            porosity = features[:, porosity_fileidx] # Ensure porosity_fileidx is defined globally or passed in
            decoded, targets = model(features, task)
            optimizer.zero_grad()
            loss = loss_fn(features, labels, targets, decoded, porosity, train_step, task)
            loss.backward()
            optimizer.step()
            # usually no scheduler step in step 1 batch loop

    elif train_step == 2:
        change_lr = True
        for task in trainloader:
            dataloader = trainloader[task]
            # ... (Keeping your freezing/unfreezing logic) ...
            
            for features, labels in dataloader:
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                features = features.to(dtype=torch.double)
                labels = labels.to(dtype=torch.double)

                labels = torch.reshape(labels, (labels.shape[0], 1))
                features = torch.reshape(features, (features.shape[0], features.shape[1]))

                optimizer.zero_grad()
                decoded, task_output = model(features, task_id=task)    
                
                loss = loss_fn(task_output, labels)
                loss.backward()
                optimizer.step()
                # REMOVED: lr_scheduler.step(loss) (Correct, do not step here)

                if task == 'sedimentation_rates':
                    train_errors['sedimentation_rates'].append(loss.item())
                elif task == 'toc':
                    train_errors['toc'].append(loss.item())

        # --- CALCULATE AVERAGE LOSS ---
        training_loss = {}
        for task in train_errors:
            if len(train_errors[task]) > 0:
                training_loss[task] = sum(train_errors[task]) / len(train_errors[task])
            else:
                training_loss[task] = None

        # --- SCHEDULER UPDATE ---
        if change_lr:
            valid_losses = []
            if training_loss['sedimentation_rates'] is not None:
                valid_losses.append(training_loss['sedimentation_rates'])
            if training_loss['toc'] is not None:
                valid_losses.append(training_loss['toc'])
            
            if valid_losses:
                avg_epoch_loss = sum(valid_losses) / len(valid_losses)
                
                # The scheduler updates its internal "patience counter" here.
                # It will NOT lower the LR unless avg_epoch_loss stops improving for 'patience' epochs.
                lr_scheduler.step(avg_epoch_loss) 

        # --- PRINT SUMMARY ---
        if (epoch+1) % 1 == 0: # Changed to 1 to see updates frequently
            # Retrieve the CURRENT learning rate from the optimizer to print it
            current_lr = optimizer.param_groups[0]['lr'] 

            if training_loss['sedimentation_rates'] is not None:
                # Fixed syntax error in f-string below
                print(f'epoch: {epoch} | Training Loss (sedimentation_rates): {training_loss["sedimentation_rates"]:.5f} | lr: {current_lr:.2e}')
            if training_loss['toc'] is not None:
                print(f'epoch: {epoch} | Training Loss (toc): {training_loss["toc"]:.5f}')

    return train_errors, training_loss
        

def validation(model, epoch, loss_fn, testloader, train_step, count, optimizer, lr_scheduler):
    model.eval()  # Set model to evaluation mode

    test_errors = {
        'sedimentation_rates': [],
        'toc': []
    }

    if train_step == 1:
        task = 'reconstruction'
        for features, labels in testloader:
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
        porosity = features[:, porosity_fileidx]

        decoded, targets = model(features, task)
        optimizer.zero_grad()
        loss = loss_fn(features, labels, targets, decoded, porosity, train_step, task)
        print(f'epoch: {epoch} | Testing Loss: {loss.item():.3f} | task_id: {task} ')
    elif train_step == 2:
        # Use dataloaders for each task, similar to training loop for new model
        for task in testloader:

            for features, labels in testloader[task]:
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                features = features.to(dtype=torch.double)
                labels = labels.to(dtype=torch.double)

                # Reshape labels and features as in trainDNN.py
                labels = torch.reshape(labels, (labels.shape[0], 1))
                features = torch.reshape(features, (features.shape[0], features.shape[1]))

                decoded, task_output = model(features, task_id=task)
                loss = loss_fn(task_output, labels)



                if task == 'sedimentation_rates':
                    test_errors['sedimentation_rates'].append(loss.item())
                elif task == 'toc':
                    test_errors['toc'].append(loss.item())

        # Calculate average test losses for each task and store in a dictionary
        testing_loss = {}
        for task in test_errors:
            if len(test_errors[task]) > 0:
                testing_loss[task] = sum(test_errors[task]) / len(test_errors[task])
            else:
                testing_loss[task] = None
        if (epoch+1)%100 == 0:
            # Print summary as in trainDNN.py
            if testing_loss['sedimentation_rates'] is not None:
                print(f'epoch: {epoch} | Testing Loss (sedimentation_rates): {testing_loss["sedimentation_rates"]:.3f}')
            if testing_loss['toc'] is not None:
                print(f'epoch: {epoch} | Testing Loss (toc): {testing_loss["toc"]:.3f}')

        # Optionally, you could append to evaluation_losses/evaluation_mae if you track them per task

        return test_errors, testing_loss 

    #print(f'epoch: {epoch} | Testing Loss: {testing_loss:.3f} | MAE: {mae:.3f} |  task_id: {task_id}')
#tensorboard
# writer = SummaryWriter()

def get_model_performance_metrics():
    return training_accuracy,  
    training_losses, 
    training_mae,
    learning_rate,
    evaluation_losses,
    evaluation_mae,
    true_labels,
    pred_labels

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
    # Normalize labels
    #labels = (labels - np.mean(labels)) / np.std(labels)

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

def create_dataloaders(train_features_dict, train_labels_dict, test_features_dict, test_labels_dict, batch_size=8, shuffle_train=True, shuffle_test=False):
    """
    Creates DataLoaders for each task from training and test arrays.

    Parameters:
    train_features_dict (dict): Dict of training features for each task, e.g. {'sedimentation_rates': ..., 'toc': ...}
    train_labels_dict (dict): Dict of training labels for each task.
    test_features_dict (dict): Dict of test features for each task.
    test_labels_dict (dict): Dict of test labels for each task.
    batch_size (int): Batch size for DataLoader.
    shuffle_train (bool): Whether to shuffle the training dataset.
    shuffle_test (bool): Whether to shuffle the test dataset.

    Returns:
    train_dataloaders (dict): Dict of DataLoaders for training data for each task.
    test_dataloaders (dict): Dict of DataLoaders for test data for each task.
    """
    train_dataloaders = {}
    test_dataloaders = {}

    for task in train_features_dict:
        # Convert to tensors
        train_features = torch.tensor(train_features_dict[task], dtype=torch.float32)
        train_labels = torch.tensor(train_labels_dict[task], dtype=torch.float32)
        test_features = torch.tensor(test_features_dict[task], dtype=torch.float32)
        test_labels = torch.tensor(test_labels_dict[task], dtype=torch.float32)

        # Create TensorDatasets
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        # Create DataLoaders
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

        train_dataloaders[task] = trainloader
        test_dataloaders[task] = testloader

        # Print summary for each task
        print(f"Task: {task}")
        print("  Train data size:", len(train_dataset))
        print("  Test data size:", len(test_dataset))

    return train_dataloaders, test_dataloaders

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

def train_model(model, loss_fn, trainloader, testloader, train_step=2, num_epochs=500):
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

    # Freeze the parameters of the decoder
    #for param in model.decoder.parameters():
    #    param.requires_grad = False
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Use a lower initial learning rate to avoid underfitting and allow finer convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # weight_decay for L2 regularization

    # Use a learning rate scheduler with more aggressive reduction to avoid overfitting
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, threshold=0.0001, threshold_mode='rel',
        cooldown=0, min_lr=1e-5, eps=1e-08
    )

    # Early stopping parameters to avoid overfitting
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # Training loop
    for epoch in range(num_epochs):
        change_lr = True  # Set to True to change learning rate
        count = 1

        # Train for one epoch
        _, training_loss = train(model, epoch, trainloader, train_step, change_lr, count, optimizer, lr_scheduler, loss_fn)
        _, testing_loss = validation(model, epoch, loss_fn, testloader, train_step, count, optimizer, lr_scheduler)

        # Early stopping logic
        if testing_loss['sedimentation_rates'] < best_val_loss - 1e-5:
            best_val_loss = testing_loss['sedimentation_rates']
            patience_counter = 0
            # Optionally save the best model here
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} to avoid overfitting.")
                break
        training_losses.append(training_loss)
        evaluation_losses.append(testing_loss)
        lr = optimizer.param_groups[0]['lr']
        learning_rate.append(lr)
        
    return  training_losses,  learning_rate,  evaluation_losses




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
            decoded, targets = model(features)
            
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