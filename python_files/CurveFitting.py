#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:02:15 2022

@author: sunms498
"""

import numpy as np
import concurrent.futures
from pathlib import Path
from scipy.optimize import curve_fit
import time
################################################################################

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_OUTPUT

filename = DATA_OUTPUT / "predictionmaps_SedRate" / "prediction_map_dist_entire.npy"
prediction_dist_CS = np.load(filename)
prediction_dist_DO = np.load(DATA_OUTPUT / "predictionmaps_SedRate" / "prediction_map_dist_entire.npy")

from utils import DATA_INTERIM
dataset_mask = np.load(DATA_INTERIM / "masks" / "mask_deepocean.npy")
dataset_mask = np.rot90(np.rot90(np.fliplr(dataset_mask)))
from utils import DATA_RAW
land_file = DATA_RAW / "island_map.npy"
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1



# Flip along the last axis (axis=2)
prediction_dist_DO = np.flip(prediction_dist_DO, axis=2)
prediction_dist_CS = np.flip(prediction_dist_CS, axis=2)

# If you also want to rotate the flipped array, you can use np.rot90
prediction_dist_DO = np.rot90(prediction_dist_DO, k=2, axes=(1, 2))
prediction_dist_CS = np.rot90(prediction_dist_CS, k=2, axes=(1, 2))



prediction_dist_overlapped = np.copy(prediction_dist_DO)

for i in range(prediction_dist_CS.shape[0]):
    # Mask the values from pm_CS where mask contains NaN values
    prediction_dist_overlapped[i, :, :][np.isnan(dataset_mask)] = prediction_dist_CS[i, :, :][np.isnan(dataset_mask)]
    if i%10 ==0:
        print(i)
for i in range(prediction_dist_overlapped.shape[0]):
    prediction_dist_overlapped[i, :, :][land_map] = np.nan


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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import tqdm
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import concurrent.futures
import time
from scipy.stats import norm
start_time = time.time()



# Define the Gaussian function using scipy.stats.norm.pdf
def gaus(x, a, x0, sigma):
    return  a*norm.pdf(x, loc=x0, scale=sigma) #check why we need scaling "a"

def curve_fit_chunk_parallel(prediction_dist_overlapped_chunk):
    nan_matrix = np.empty((3, 3))
    nan_matrix.fill(np.nan)
    # Initialize lists to store optimized parameters and covariance matrices for each point
    param_opt_all = np.empty([3, prediction_dist_overlapped_chunk.shape[1], prediction_dist_overlapped_chunk.shape[2]])
    cov_matrix_all = np.empty([3, 3, prediction_dist_overlapped_chunk.shape[1], prediction_dist_overlapped_chunk.shape[2]])
    tolerance_all = np.empty([prediction_dist_overlapped_chunk.shape[1], prediction_dist_overlapped_chunk.shape[2]])
    failed_fits = []
    failed_fits_num = 0
    def fit_gaussian_serial(i, j):
            x_data = prediction_dist_overlapped_chunk[:, i, j]

            if np.isnan(x_data).any():
                param_opt_all[:, i, j] = np.array([np.nan, np.nan, np.nan])
                cov_matrix_all[:, :, i, j] = nan_matrix
                # failed_fits.append((i, j, str(e)))
                return 

            hist, bin_edges = np.histogram(x_data, bins=500) # bins affect the curve that we obtain, but it mainly affects the height of the curve
            hist = hist / sum(hist) #doesnt this normalise the histograms, with the sum of the histograms to 1?

            n = len(hist)
            x_hist = np.zeros((n), dtype=float)
            for ii in range(n):
                x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

            y_hist = hist
            mean = np.mean(x_data)#sum(x_hist * y_hist) / sum(y_hist)
            sigma = np.std(x_data)#sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
            #if sigma<0:
            #    print(sigma)
            sigma_squared = sigma **2
            tolerance = 1e-10  # Initial tolerance value
            max_iterations = 10  # Maximum number of iterations
            iteration = 0

            
            while iteration < max_iterations:
                try:
                    # Gaussian least-square fitting process
                    param_optimised, param_covariance_matrix = curve_fit(
                        gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000, ftol=tolerance
                    )

                    # If the fit is successful, break out of the loop
                    break

                except Exception as e:
                    # If the fit fails, increase the tolerance for the next iteration
                    tolerance *= 10
                    iteration += 1
            
            if iteration == max_iterations:
                # If no successful fit is found, handle it as needed
                param_opt_all[:, i, j] = np.array([np.nan, np.nan, np.nan])
                cov_matrix_all[:, :, i, j] = nan_matrix
                failed_fits.append((i, j, "Fit failed even after increasing tolerance"))
            else:
                # Append the successful results to the lists
                param_opt_all[:, i, j] = param_optimised
                cov_matrix_all[:, :, i, j] = param_covariance_matrix
                tolerance_all[i, j] = tolerance
        



    for i in range(prediction_dist_overlapped_chunk.shape[1]):
        for j in range(prediction_dist_overlapped_chunk.shape[2]):
            fit_gaussian_serial(i, j)

    return param_opt_all, cov_matrix_all, tolerance_all, failed_fits, failed_fits_num

param_opt_list, cov_matrix_list, tolerance_list, failed_fits_list, failed_fits_num_list = [], [], [], [], []
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(curve_fit_chunk_parallel, chunks))

# Extract the results from the list
for result in results:
    param_opt_list.append(result[0])
    cov_matrix_list.append(result[1])
    tolerance_list.append(result[2])
    failed_fits_list.append(result[3])
    failed_fits_num_list.append(result[4])
    
end_time = time.time()    

print(param_opt_list[3][1,:,:])
param_opt_list = np.concatenate(param_opt_list, axis=1)
print(param_opt_list.shape)

tolerance_list = np.concatenate(tolerance_list, axis=0)
print(tolerance_list.shape)

cov_matrix_list = np.concatenate(cov_matrix_list, axis=2)
print(cov_matrix_list.shape)

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
np.save("preprocessed/param_opt_list", param_opt_list)
np.save("preprocessed/cov_matrix_list", cov_matrix_list)
np.save("preprocessed/tolerance_list", tolerance_list)