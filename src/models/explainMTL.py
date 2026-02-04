import shap
import pandas as pd
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def explain_shap(model, features, task_id="sedimentation_rates",
                     feature_names=None, explainer_path='explainer.pkl',
                     shap_values_path='shap_values.pkl',
                     force_plot_index=0, summary_max_display=10,
                     output_path='shap_summary_plot.png'):

    # ✅ Wrap the model for the specific task
    class TaskWrapper(torch.nn.Module):
        def __init__(self, mtl_model, task_id):
            super().__init__()
            self.mtl_model = mtl_model
            self.task_id = task_id

        def forward(self, x):
            _, task_output = self.mtl_model(x, task_id=self.task_id)
            return task_output

    # Wrap the MTL model for a specific task
    wrapped_model = TaskWrapper(model, task_id="sedimentation_rates")
    wrapped_model.eval()

    # Convert input features to numpy for Kernel SHAP
    X = features.astype(np.float64)
    features = features[:200,:]
    # Select a background subset for efficiency
    background = X[:min(100, len(X))]

    # Create Kernel SHAP explainer
    explainer = shap.KernelExplainer(
        lambda x: wrapped_model(torch.tensor(x, dtype=torch.float64)).detach().cpu().numpy(),
        background
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    # If output is a list (e.g., one element for each output dimension)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.initjs()

    #features_np = features_tensor.cpu().detach().numpy()
    features_np = X

    """
    # Save SHAP explainer and values
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)
    with open(shap_values_path, 'wb') as f:
        pickle.dump(shap_values, f)
    """
        # Create a summary plot
    plt.figure(dpi=300)
    shap.summary_plot(shap_values, features_np, max_display=summary_max_display, show=False)
    print(shap_values)
    # Save SHAP values as a numpy array
    np.save(str(output_path).replace('.png', '_shap_values.npy'), shap_values)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ SHAP analysis complete for task: {task_id}")
    print(f"Results saved to {output_path}")

    return shap_values
"""
# ✅ Wrap the model for the specific task class TaskWrapper(torch.nn.Module): def __init__(self, mtl_model, task_id): super().__init__() self.mtl_model = mtl_model self.task_id = task_id def forward(self, x): _, task_output = self.mtl_model(x, task_id=self.task_id) return task_output wrapped_model = TaskWrapper(model, task_id) wrapped_model.eval() # Convert the input features to a PyTorch tensor with the correct data type features_tensor = torch.tensor(features, dtype=torch.float64) if torch.cuda.is_available(): wrapped_model = wrapped_model.cuda() features_tensor = features_tensor.cuda() # ✅ Create the SHAP DeepExplainer for the single-task wrapper background = features_tensor[:min(100, len(features_tensor))] # subset for efficiency explainer = shap.DeepExplainer(wrapped_model, background) # Calculate SHAP values for all instances shap_values = explainer.shap_values(features_tensor)"""