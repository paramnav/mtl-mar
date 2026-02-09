import shap
import pandas as pd
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def explain_shap(model, features, feature_names=None, explainer_path='explainer.pkl', shap_values_path='shap_values.pkl', force_plot_index=0, summary_max_display=10, output_path='shap_summary_plot.png'):
    # Convert the input features to a PyTorch tensor with the correct data type
    features = torch.tensor(features, dtype=torch.float64)

    if torch.cuda.is_available():
        features = features.cuda()

    # Create a DeepExplainer for the PyTorch model
    explainer = shap.DeepExplainer(model, features)

    # Calculate SHAP values for all instances with a progress bar
    # If features is a 2D tensor, we can iterate over rows and collect shap values per instance
    # However, DeepExplainer can take a batch, but to show progress, we process in batches
    features_np = features.cpu().detach().numpy()
    batch_size = 64  # You can adjust this batch size as needed
    num_samples = features_np.shape[0]
    shap_values_list = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Calculating SHAP values"):
        batch = features[i:i+batch_size]
        batch_shap_values = explainer.shap_values(batch)
        # If shap_values is a list (for multi-output), handle accordingly
        if isinstance(batch_shap_values, list):
            if len(shap_values_list) == 0:
                shap_values_list = [[] for _ in range(len(batch_shap_values))]
            for j in range(len(batch_shap_values)):
                shap_values_list[j].append(batch_shap_values[j])
        else:
            shap_values_list.append(batch_shap_values)

    # Concatenate the results
    if isinstance(shap_values_list, list) and len(shap_values_list) > 0 and isinstance(shap_values_list[0], list):
        # Multi-output case
        shap_values = []
        for j in range(len(shap_values_list)):
            shap_values.append(np.concatenate(shap_values_list[j], axis=0))
    elif isinstance(shap_values_list, list):
        shap_values = np.concatenate(shap_values_list, axis=0)
    else:
        shap_values = shap_values_list

    # init the JS visualization code
    shap.initjs()

    # Assuming features is a NumPy array
    features_np = features.cpu().detach().numpy()
    """
    # Create a force plot for the specified instance
    if force_plot_index < len(features_np):
        shap.force_plot(explainer.expected_value[0], shap_values[0][force_plot_index], features_np[force_plot_index], feature_names=feature_names)
    """
    # Create a summary plot
    plt.figure(dpi=300)
    shap.summary_plot(shap_values, features_np, max_display=summary_max_display, show=False)
    print(shap_values)
    # Save SHAP values as a numpy array
    np.save(str(output_path).replace('.png', '_shap_values.npy'), shap_values)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    #"""
        # Save the explainer and SHAP values using pickle
    with open(explainer_path, 'wb') as explainer_file:
        pickle.dump(explainer, explainer_file)

    with open(shap_values_path, 'wb') as shap_values_file:
        pickle.dump(shap_values, shap_values_file)
    #"""
