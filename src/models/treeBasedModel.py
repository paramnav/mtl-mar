import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import xgboost as xgb
import catboost as cb

def train_tree_based_model(features_train, labels_train, features_test, labels_test, model_type="random_forest", n_estimators=100, max_depth=None, random_seed=42):
    # Initialize the model based on type
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_seed,
            verbosity=0,
            n_jobs=-1,
            learning_rate=0.01
        )
    elif model_type == "catboost":
        model = cb.CatBoostRegressor(
            iterations=n_estimators,
            random_seed=random_seed,
            verbose=0,
            learning_rate=0.01,
            loss_function="RMSE"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'random_forest', 'xgboost', 'catboost'.")

    # Train the model
    model.fit(features_train, labels_train)

    # Make predictions
    train_predictions = model.predict(features_train)
    test_predictions = model.predict(features_test)

    # Calculate MAE
    train_mae = mean_absolute_error(labels_train, train_predictions)
    test_mae = mean_absolute_error(labels_test, test_predictions)

    return model, train_mae, test_mae, train_predictions, test_predictions

import seaborn as sns

def plot_tree_model_performance(n_estimators_list,
                                train_mae_list, test_mae_list,
                                train_r2_list, test_r2_list,
                                train_corr_list, test_corr_list):
    """
    Plots MAE, R², and Correlation metrics vs. number of estimators for tree-based models.
    All metrics are plotted with a logarithmic x-axis.
    The three plots are stacked vertically and share the x-axis.
    Smooth lines are not interpolated, just lines through the points.
    The best model is highlighted as a red point.
    """
    sns.set(style="whitegrid", context="talk")
    n_estimators = np.array(n_estimators_list)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # --- 1. MAE Plot ---
    ax = axes[0]
    ax.plot(n_estimators, train_mae_list, label='Train MAE', color='C0')
    ax.plot(n_estimators, test_mae_list, label='Test MAE', color='C1')
    # Highlight minimum test MAE
    min_test_mae_idx = int(np.argmin(test_mae_list))
    ax.scatter(n_estimators[min_test_mae_idx], test_mae_list[min_test_mae_idx], 
               color='red', s=80, zorder=5, label='Best Test MAE')
    ax.annotate(f'{test_mae_list[min_test_mae_idx]:.3f}',
                (n_estimators[min_test_mae_idx], test_mae_list[min_test_mae_idx]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=11)
    ax.set_ylabel('MAE')
    ax.legend(frameon=True, loc='best')
    ax.set_xscale('log')

    # --- 2. R² Plot ---
    ax = axes[1]
    ax.plot(n_estimators, train_r2_list, label='Train $R^2$', color='C0')
    ax.plot(n_estimators, test_r2_list, label='Test $R^2$', color='C1')
    # Highlight maximum test R2
    max_test_r2_idx = int(np.argmax(test_r2_list))
    ax.scatter(n_estimators[max_test_r2_idx], test_r2_list[max_test_r2_idx], 
               color='red', s=80, zorder=5, label='Best Test $R^2$')
    ax.annotate(f'{test_r2_list[max_test_r2_idx]:.3f}',
                (n_estimators[max_test_r2_idx], test_r2_list[max_test_r2_idx]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=11)
    ax.set_ylabel('$R^2$')
    ax.legend(frameon=True, loc='best')
    ax.set_xscale('log')

    # --- 3. Correlation Plot ---
    ax = axes[2]
    ax.plot(n_estimators, train_corr_list, label='Train Corr.', color='C0')
    ax.plot(n_estimators, test_corr_list, label='Test Corr.', color='C1')
    # Highlight maximum test correlation
    max_test_corr_idx = int(np.argmax(test_corr_list))
    ax.scatter(n_estimators[max_test_corr_idx], test_corr_list[max_test_corr_idx], 
               color='red', s=80, zorder=5, label='Best Test Corr.')
    ax.annotate(f'{test_corr_list[max_test_corr_idx]:.3f}',
                (n_estimators[max_test_corr_idx], test_corr_list[max_test_corr_idx]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=11)
    ax.set_ylabel('Correlation')
    ax.legend(frameon=True, loc='best')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Estimators')

    plt.tight_layout()
    plt.show()

    
def create_scatter_plot(true_values, predictions, output_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Random Forest Predictions vs True Values')
    
    # Calculate and display statistics
    mae = mean_absolute_error(true_values, predictions)
    corr, _ = pearsonr(true_values, predictions)
    r2 = corr**2
    
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nCorrelation: {corr:.4f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.savefig(output_path)
    plt.close()
    
    return mae, corr, r2

# Calculate correlation between features and target
def calculate_correlations(features, labels):
    correlations = []
    for i in range(features.shape[1]):
        corr, _ = pearsonr(features[:, i], labels)
        correlations.append(corr)
    return correlations

# Plotting function with metrics
def plot_predictions_density(y_train, train_preds, y_test, test_preds, model_name, n_estimators):
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0,1]
    test_corr = np.corrcoef(y_test, test_preds)[0,1]
    
    # Scatter plot
    plt.figure(figsize=(7,7))
    plt.scatter(y_train, train_preds, color='blue', label=f'Train (MAE={train_mae:.3f}, CC={train_corr:.3f})', alpha=0.6)
    plt.scatter(y_test, test_preds, color='orange', label=f'Test (MAE={test_mae:.3f}, CC={test_corr:.3f})', alpha=0.6)
    max_val = max(y_train.max(), y_test.max(), train_preds.max(), test_preds.max())
    min_val = min(y_train.min(), y_test.min(), train_preds.min(), test_preds.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f"{model_name} Predictions (n_estimators={n_estimators:.0f})")
    plt.legend()
    plt.grid(True)
    
    # Add a text box for metrics
    #textstr = f"Train MAE: {train_mae:.4f}\nTrain corr coeff: {train_corr:.4f}\nTest MAE: {test_mae:.4f}\nTest corr coeff: {test_corr:.4f}"
    #plt.gcf().text(0.65, 0.15, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()


