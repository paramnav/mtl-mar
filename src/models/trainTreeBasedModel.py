import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns


def train_tree_model(X_train, y_train, X_test, y_test, model_type="xgboost", n_estimators=100, random_seed=42, **kwargs):
    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_seed,
            verbosity=0,
            n_jobs=-1,
            **kwargs
        )
    elif model_type == "catboost":
        model = cb.CatBoostRegressor(
            iterations=n_estimators,
            random_seed=random_seed,
            verbose=0,
            **kwargs
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'xgboost', 'catboost', or 'random_forest'.")

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    return model, train_mae, test_mae, train_preds, test_preds


def plot_predictions(y_train, train_preds, y_test, test_preds, model_name, n_estimators):
    sns.set(style='whitegrid', context='notebook')
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0,1]
    test_corr = np.corrcoef(y_test, test_preds)[0,1]

    # Combine data
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train, train_preds, color='dodgerblue', alpha=0.6, edgecolor='k',
                label=f'Train (MAE={train_mae:.3f}, r={train_corr:.3f})')
    plt.scatter(y_test, test_preds, color='darkorange', alpha=0.6, edgecolor='k',
                label=f'Test (MAE={test_mae:.3f}, r={test_corr:.3f})')

    # Identity line
    all_vals = np.concatenate([y_train, y_test, train_preds, test_preds])
    min_val, max_val = all_vals.min(), all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    # Axis and labels
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name} Predictions (n_estimators={n_estimators})', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.axis('equal')

    # Metrics box
    metrics_text = (
        f"Train MAE: {train_mae:.3f}\n"
        f"Train r: {train_corr:.3f}\n"
        f"Test MAE: {test_mae:.3f}\n"
        f"Test r: {test_corr:.3f}"
    )
    plt.gcf().text(0.65, 0.15, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()
    
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
            print(f"Duplicate row with index {row_idx} was in group {group_idx} and was replaced with NaN values")

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

def create_train_test_split(features, labels, test_size_ratio=0.3, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size_ratio, 
        shuffle=True, 
        random_state=42
    )


    return trainloader, testloader