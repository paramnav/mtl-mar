"""
Fit Gaussian distributions to prediction uncertainty distributions.

This script loads prediction distributions from Monte Carlo dropout,
fits Gaussian curves to each grid cell's distribution, and saves
the fitted parameters (mean, std, amplitude) and covariance matrices.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import concurrent.futures
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
import time
import sys

# Get project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_OUTPUT, DATA_INTERIM, DATA_RAW

# Load prediction distributions
filename = DATA_OUTPUT / "predictionmaps_SedRate" / "prediction_map_dist_entire.npy"
prediction_dist_CS = np.load(filename)
prediction_dist_DO = np.load(filename)

# Load masks
dataset_mask = np.load(DATA_INTERIM / "masks" / "mask_deepocean.npy")
dataset_mask = np.rot90(np.rot90(np.fliplr(dataset_mask)))

land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1

# Flip and rotate prediction distributions
prediction_dist_DO = np.flip(prediction_dist_DO, axis=2)
prediction_dist_CS = np.flip(prediction_dist_CS, axis=2)
prediction_dist_DO = np.rot90(prediction_dist_DO, k=2, axes=(1, 2))
prediction_dist_CS = np.rot90(prediction_dist_CS, k=2, axes=(1, 2))

# Overlap CS and DO predictions
prediction_dist_overlapped = np.copy(prediction_dist_DO)
for i in range(prediction_dist_CS.shape[0]):
    prediction_dist_overlapped[i, :, :][np.isnan(dataset_mask)] = (
        prediction_dist_CS[i, :, :][np.isnan(dataset_mask)]
    )
    if i % 10 == 0:
        print(f"Processing distribution {i}/{prediction_dist_CS.shape[0]}")

# Apply land mask
for i in range(prediction_dist_overlapped.shape[0]):
    prediction_dist_overlapped[i, :, :][land_map] = np.nan

# Split into chunks for parallel processing
prediction_dist_overlapped_shape = prediction_dist_overlapped.shape
num_chunks = 360
chunk_size = prediction_dist_overlapped_shape[1] // num_chunks
chunks_list = []

for i in range(num_chunks):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size
    chunk = prediction_dist_overlapped[:, start_index:end_index, :]
    chunks_list.append(chunk)

chunks = np.array(chunks_list)

# Define the Gaussian function
def gaus(x, a, x0, sigma):
    """Gaussian function for curve fitting."""
    return a * norm.pdf(x, loc=x0, scale=sigma)

def curve_fit_chunk_parallel(prediction_dist_overlapped_chunk):
    """
    Fit Gaussian distributions to prediction distributions for a chunk.
    
    Parameters:
    -----------
    prediction_dist_overlapped_chunk : np.ndarray
        Chunk of prediction distributions with shape (n_samples, lat, lon)
    
    Returns:
    --------
    param_opt_all : np.ndarray
        Optimized parameters [amplitude, mean, std] for each grid cell
    cov_matrix_all : np.ndarray
        Covariance matrices for each grid cell
    tolerance_all : np.ndarray
        Tolerance values used for each fit
    failed_fits : list
        List of failed fits with coordinates
    failed_fits_num : int
        Number of failed fits
    """
    nan_matrix = np.empty((3, 3))
    nan_matrix.fill(np.nan)
    
    param_opt_all = np.empty([
        3, prediction_dist_overlapped_chunk.shape[1], 
        prediction_dist_overlapped_chunk.shape[2]
    ])
    cov_matrix_all = np.empty([
        3, 3, prediction_dist_overlapped_chunk.shape[1], 
        prediction_dist_overlapped_chunk.shape[2]
    ])
    tolerance_all = np.empty([
        prediction_dist_overlapped_chunk.shape[1], 
        prediction_dist_overlapped_chunk.shape[2]
    ])
    failed_fits = []
    failed_fits_num = 0
    
    def fit_gaussian_serial(i, j):
        """Fit Gaussian to distribution at grid cell (i, j)."""
        x_data = prediction_dist_overlapped_chunk[:, i, j]
        
        if np.isnan(x_data).any() or len(x_data) == 0:
            param_opt_all[:, i, j] = np.array([np.nan, np.nan, np.nan])
            cov_matrix_all[:, :, i, j] = nan_matrix
            return
        
        # Create histogram
        hist, bin_edges = np.histogram(x_data, bins=500)
        hist = hist / sum(hist)  # Normalize
        
        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for ii in range(n):
            x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2
        
        y_hist = hist
        
        # Initial parameter estimates
        mean = np.mean(x_data)
        sigma = np.std(x_data)
        
        tolerance = 1e-10  # Initial tolerance value
        max_iterations = 10  # Maximum number of iterations
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Gaussian least-square fitting
                param_optimised, param_covariance_matrix = curve_fit(
                    gaus, x_hist, y_hist, 
                    p0=[max(y_hist), mean, sigma], 
                    maxfev=5000, 
                    ftol=tolerance
                )
                break
            except Exception:
                tolerance *= 10
                iteration += 1
        
        if iteration == max_iterations:
            param_opt_all[:, i, j] = np.array([np.nan, np.nan, np.nan])
            cov_matrix_all[:, :, i, j] = nan_matrix
            failed_fits.append((i, j, "Fit failed even after increasing tolerance"))
        else:
            param_opt_all[:, i, j] = param_optimised
            cov_matrix_all[:, :, i, j] = param_covariance_matrix
            tolerance_all[i, j] = tolerance
    
    # Fit Gaussian to each grid cell
    for i in range(prediction_dist_overlapped_chunk.shape[1]):
        for j in range(prediction_dist_overlapped_chunk.shape[2]):
            fit_gaussian_serial(i, j)
    
    return param_opt_all, cov_matrix_all, tolerance_all, failed_fits, failed_fits_num

# Process chunks in parallel
start_time = time.time()
print("Starting parallel curve fitting...")

param_opt_list, cov_matrix_list, tolerance_list, failed_fits_list, failed_fits_num_list = (
    [], [], [], [], []
)

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(curve_fit_chunk_parallel, chunks))

# Extract the results
for result in results:
    param_opt_list.append(result[0])
    cov_matrix_list.append(result[1])
    tolerance_list.append(result[2])
    failed_fits_list.append(result[3])
    failed_fits_num_list.append(result[4])

end_time = time.time()
elapsed_time = end_time - start_time

# Concatenate results
param_opt_list = np.concatenate(param_opt_list, axis=1)
tolerance_list = np.concatenate(tolerance_list, axis=0)
cov_matrix_list = np.concatenate(cov_matrix_list, axis=2)

print(f"Fitted parameters shape: {param_opt_list.shape}")
print(f"Tolerance list shape: {tolerance_list.shape}")
print(f"Covariance matrix shape: {cov_matrix_list.shape}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Save results
output_dir = Path("preprocessed")
output_dir.mkdir(exist_ok=True)

np.save(output_dir / "param_opt_list.npy", param_opt_list)
np.save(output_dir / "cov_matrix_list.npy", cov_matrix_list)
np.save(output_dir / "tolerance_list.npy", tolerance_list)

print(f"Results saved to {output_dir}/")
print(f"Total failed fits: {sum(len(f) for f in failed_fits_list)}")
