#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:02:15 2022

@author: sunms498
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from os import walk
import pickle
from pathlib import Path
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

import numpy as np

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
import numpy as np

"""
# Adding colorbar
cbar = plt.colorbar(label='Values (log scale)')
cbar.ax.tick_params(labelsize=14)

# Adding labels and title
plt.xlabel('X-axis', fontsize=16)
plt.ylabel('Y-axis', fontsize=16)
plt.title('Log-scaled Image', fontsize=18)

# Show the plot
plt.show()
"""
from pathlib import Path
from matplotlib.colors import LogNorm
################################################################################
#"""
#filename = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/prediction_map_TOC_noconstraint_supervised_men_CS_constraint05.npy"
filename = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/prediction_map_dist_CS.npy"
prediction_dist_CS = np.load(filename)
prediction_dist_DO = np.load("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/prediction_map_dist_DO.npy")




uncertainty_map_CS = np.var(prediction_dist_CS, axis = 0)
prediction_map_CS = np.mean(prediction_dist_CS, axis = 0)

uncertainty_map_DO = np.var(prediction_dist_DO, axis = 0)
prediction_map_DO = np.mean(prediction_dist_DO, axis = 0)

uncertainty_map_CS = np.rot90(np.rot90(np.fliplr(uncertainty_map_CS)))
prediction_map_CS = np.rot90(np.rot90(np.fliplr(prediction_map_CS)))


uncertainty_map_DO = np.rot90(np.rot90(np.fliplr(uncertainty_map_DO)))
prediction_map_DO = np.rot90(np.rot90(np.fliplr(prediction_map_DO)))


#filename = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/uncertainty_map_TOC_noconstraint_supervised_men_CS_constraint05.npy"
#uncertainty_map_CS = np.load(filename)
#uncertainty_map_DO = np.load("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/uncertainty_map_TOC_noconstraint_supervised_men_DO.npy")



dataset_mask = np.load("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/masks/mask_deepocean.npy")
dataset_mask = np.rot90(np.rot90(np.fliplr(dataset_mask)))

# Create a copy of pm_CS
pm_overlapped = np.copy(prediction_map_DO)

# Mask the values from pm_CS where mask contains NaN values
pm_overlapped[np.isnan(dataset_mask)] = prediction_map_CS[np.isnan(dataset_mask)]

# Create a copy of pm_CS
um_overlapped = np.copy(uncertainty_map_DO)

# Mask the values from pm_CS where mask contains NaN values
um_overlapped[np.isnan(dataset_mask)] = uncertainty_map_CS[np.isnan(dataset_mask)]


#prediction_dist_overlapped = np.copy(prediction_dist_DO)
#pm_overlapped[np.isnan(dataset_mask)] = prediction_map_CS[np.isnan(dataset_mask)]



"""
pm_overlapped = np.load("/home/sunms498/work/toc-neural-nets/SelectFeaturesRun/predictionmap_parametersharing/prediction_map_parametersharing_sedrate_soft_lambda_06.npy")
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#plt.figure(figsize=[25, 15], dpi=200)

# Create a lognorm instance
#lognorm = colors.LogNorm(vmin=0.01, vmax=800)

# Plot the overlapped array with lognorm
#plt.imshow(pm_overlapped, norm = lognorm,  cmap="turbo")
#cbar = plt.colorbar()
##################################################################################

# Add colorbar
#cbar = plt.colorbar()

#plt.show()

ds=2.6

file_porosity = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/porosity/grl53425-sup-0002-supinfo.grd"
#file_porosity = "file_porosity = "/home/sunms498/Data/paper_porosity/GPSM_Restreppo/prediction.nc""
rootgrp_porosity = xr.open_dataset(Path(file_porosity), format = 'NETCDF4')
                                                                   
dataset_porosity = rootgrp_porosity.variables['z']

dataset_numpy_porosity = dataset_porosity.to_numpy()
dataset_numpy_porosity = np.rot90(np.rot90(np.fliplr(dataset_numpy_porosity)))
"""

file_2020 = "/home/sunms498/Data/paper2020/GPSM_Restreppo/prediction.nc"
#file_porosity = "file_porosity = "/home/sunms498/Data/paper_porosity/GPSM_Restreppo/prediction.nc""
rootgrp_2020 = xr.open_dataset(Path(file_2020), format = 'NETCDF4')


file_2021 = "/home/sunms498/Data/paper2021/GPSM_Restreppo/Log10_prediction_5m.nc"
#file_porosity = "file_porosity = "/home/sunms498/Data/paper_porosity/GPSM_Restreppo/prediction.nc""
rootgrp_2021 = xr.open_dataset(Path(file_2021), format = 'NETCDF4')


latitudes = rootgrp_porosity.variables['y'] # in theory all the same
longitudes = rootgrp_porosity.variables['x']

latitudes2020 = rootgrp_2020.variables['lat'] # in theory all the same
longitudes2020 = rootgrp_2020.variables['lon']

latitudes2021 = rootgrp_2021.variables['lat'] # in theory all the same
longitudes2021 = rootgrp_2021.variables['lon']

dataset_porosity = rootgrp_porosity.variables['z']
dataset_2020 = rootgrp_2020.variables['z']
dataset_2021 = rootgrp_2021.variables['z']


# #plt.figure(figsize=([20,10]), dpi=300);plt.imshow(np.rot90(np.rot90(np.fliplr(dataset_numpy))))
dataset_numpy_porosity = dataset_porosity.to_numpy()
dataset_numpy_porosity = np.rot90(np.rot90(np.fliplr(dataset_numpy_porosity)))

dataset_numpy_2020 = dataset_2020.to_numpy()
dataset_numpy_2020 = np.rot90(np.rot90(np.fliplr(dataset_numpy_2020)))

dataset_numpy_2021 = dataset_2021.to_numpy()
dataset_numpy_2021 = np.rot90(np.rot90(np.fliplr(dataset_numpy_2021)))

dataset2020mar = (1-dataset_numpy_porosity/100)*ds*(10**dataset_numpy_2020)##


def find_nearest(value,vector):
    vector = np.float64(vector)-np.float64(value);
    return np.argmin(np.abs(vector))


# data_path = Path(Path().resolve().parents[2] / "Data" / "RestrepoKNNsSed") 
# sed_labels_file = "sediments_all_updated.xlsx"

# df = pd.read_excel(Path(data_path / sed_labels_file))

# df = df[["Latitude","Longitude","Sedimentation rate [cm/ka]"]]
# df = df[df["Longitude"]>-180]
# df = df.dropna(how="any")
# df = df[df["Sedimentation rate [cm/ka]"]<200]

# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
# gdf = GeoDataFrame(df, geometry=geometry)  
# fig, ax=plt.subplots(figsize=(20,10), dpi=300)
# gdf.plot(ax = world.plot(ax=ax), column = df["Sedimentation rate [cm/ka]"], vmin = 0, vmax = 200,legend = True, legend_kwds={"label":"{Sedimentation rate [cm/ka]}"});
# fig.tight_layout(pad=0)
# ax.set_xlim(-180,180)
# ax.set_ylim(-90,90)
# fig.savefig("/home/sunms498/work/Img/sedimentation_labels.png", bbox_inches='tight', pad_inches=0)
# df.describe()

"""
#feature_image_path ="../../../Data/LeeKNNsTOC/RestrepoFeatureMaps/"
restrepo_features_path = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/GL_LATITUDE_DD.5m.nc"
#restrepo_features_path_flipped = Path("/media/sunms498/EXTERNAL_USB/Features_Phrampus_flipped")
restrepo_files = []

for (dirpath, dirnames, filenames) in walk(restrepo_features_path):
    restrepo_files.extend(filenames)

restrepo_files.sort()
len(restrepo_files)

rootgrp = xr.open_dataset(restrepo_features_path, format = 'NETCDF4')
latitudes = rootgrp.variables['lat']
longitudes = rootgrp.variables['lon']

# restrepo_features = np.empty([len(df),len(restrepo_files[0:len(restrepo_files)])])
# restrepo_features[:] = np.nan
# restrepo_X_mean = np.empty(0)
# restrepo_X_std = np.empty(0)

file_idx = -1
corrupted_files = 0


 
rootgrp = xr.open_dataset(restrepo_features_path, format = 'NETCDF4')


    
#     latitudes = rootgrp.variables['lats'] # in theory all the same
#     longitudes = rootgrp.variables['lons']

dataset = rootgrp.variables['data']
area = 9.297**2 * np.cos(dataset/180 * np.pi)




land_file = Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/island_map.npy")
land_map = np.load(land_file)
land_map[np.isnan(land_map)] = 1


pm_overlapped[land_map] = 0
um_overlapped[land_map] = 0
"""
# prediction_map = np.load("prediction_map(withoutlog)RestrepoFeats__AllVS0.2.npy")
# prediction_map_flipped= np.rot90(np.rot90(np.fliplr(prediction_map)))
# prediction_map_flipped[land_map]=np.nan

# mar_naveen = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_flipped/1e3)

#prediction_map_nonan= np.load("Sed_regresion_083_RestrepoFeats_AllVS0.2_noNaN.npy")
prediction_map_nonan= pm_overlapped#np.load("/home/sunms498/work/toc-neural-nets/SelectFeaturesRun/notebooks/SedRate_Naveen/pm_overlap.npy")
#prediction_map_flipped_nonan= np.rot90(np.rot90(np.fliplr(prediction_map_nonan)))
prediction_map_flipped_nonan = prediction_map_nonan
prediction_map_flipped_nonan[land_map]=np.nan

mar_naveen_nonan = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_flipped_nonan/1e3)
# mar_naveen = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_flipped/1e3)

# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction from neural network model(ds=2.6)")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log

"""


# Flip along the last axis (axis=2)
prediction_dist_DO = np.flip(prediction_dist_DO, axis=2)
prediction_dist_CS = np.flip(prediction_dist_CS, axis=2)

# If you also want to rotate the flipped array, you can use np.rot90
prediction_dist_DO = np.rot90(prediction_dist_DO, k=2, axes=(1, 2))
prediction_dist_CS = np.rot90(prediction_dist_CS, k=2, axes=(1, 2))



prediction_dist_overlapped = np.copy(prediction_dist_DO)

for i in range(prediction_dist_overlapped.shape[0]):
    prediction_dist_overlapped[i, :, :][land_map] = np.nan


for i in range(prediction_dist_CS.shape[0]):
    # Mask the values from pm_CS where mask contains NaN values
    prediction_dist_overlapped[i, :, :][np.isnan(dataset_mask)] = prediction_dist_CS[i, :, :][np.isnan(dataset_mask)]
    if i%10 ==0:
        print(i)
# -*- coding: utf-8 -*-
"""
@author: wasy
"""
#STEP 1: IMPORT LIBRARIES ----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

#close all opened figure
plt.close("all")


# Initialize lists to store optimized parameters and covariance matrices for each point
param_opt_all = np.zeros([3, prediction_dist_overlapped.shape[1], prediction_dist_overlapped.shape[2]])
cov_matrix_all = np.zeros([3, 3, prediction_dist_overlapped.shape[1], prediction_dist_overlapped.shape[2]])
failed_fits = []
failed_fits_num = 0


# Given covariance matrix shape
covariance_matrix_shape = (3, 3)

# Create a NaN matrix of the same shape
nan_matrix = np.full(covariance_matrix_shape, np.nan)


#Calculating the Gaussian PDF values given Gaussian parameters and random variable X
def gaus(X,C,X_mean,sigma):
    return C*exp(-(X-X_mean)**2/(2*sigma**2))


# Iterate over each point in the matrix
for i in range(prediction_dist_overlapped.shape[1]):
    print(i)
    for j in range(prediction_dist_overlapped.shape[2]):
        x_data = prediction_dist_overlapped[:, i, j]

        if np.isnan(x_data).any():
            param_opt_all[:, i, j] = np.array([np.nan,np.nan, np.nan])
            cov_matrix_all[:, :, i, j] = nan_matrix
            #failed_fits.append((i, j, str(e)))
            continue
        


        hist, bin_edges = np.histogram(x_data, bins = 100)
        hist=hist/sum(hist)

        n = len(hist)
        x_hist=np.zeros((n),dtype=float) 
        for ii in range(n):
            x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2

        y_hist=hist
        mean = sum(x_hist*y_hist)/sum(y_hist)                  
        sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 
        try:
            # Gaussian least-square fitting process
            param_optimised, param_covariance_matrix = curve_fit(
                gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000
            )
            
            
            # Append the results to the lists
            param_opt_all[:, i, j] = param_optimised
            cov_matrix_all[:, :, i, j] = param_covariance_matrix

        except Exception as e:
            
            print(f"Failed fit at ({i}, {j}) - Number: {failed_fits_num}")
            failed_fits_num = failed_fits_num + 1
            # Handle the case where fitting fails
            param_opt_all[:, i, j] = np.array([np.nan,np.nan, np.nan])
            cov_matrix_all[:, :, i, j] = nan_matrix
            failed_fits.append((i, j, str(e)))
        
        # Append the results to the lists
        #param_opt_all.append(param_optimised)
        #cov_matrix_all.append(param_covariance_matrix)        

"""
from multiprocessing import Pool

def process_point(args):
    i, j, x_data = args
    try:
        hist, bin_edges = np.histogram(x_data)
        hist = hist / sum(hist)

        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for ii in range(n):
            x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

        y_hist = hist
        mean = sum(x_hist * y_hist) / sum(y_hist)
        sigma = sum(y_hist * (x_hist - mean)) / sum(y_hist)

        # Gaussian least-square fitting process
        param_optimised, param_covariance_matrix = curve_fit(
            gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000
        )

        return param_optimised, param_covariance_matrix
    except Exception as e:
        print("heyy!!")
        return np.array([np.nan, np.nan, np.nan]), np.full(covariance_matrix_shape, np.nan)
        #failed_fits.append((i, j))
# Prepare a list of arguments for each point
point_args = [(i, j, prediction_dist_overlapped[:, i, j])
              for i in range(prediction_dist_overlapped.shape[1])
              for j in range(prediction_dist_overlapped.shape[2])]

# Create a Pool with the number of processes you want
with Pool() as pool:
    # Use map to apply the process_point function to each set of arguments in parallel
    results = pool.map(process_point, point_args)

# Extract the results
param_opt_list, cov_matrix_list = zip(*results)
        


"""

#print fit Gaussian parameters
print("Fit parameters: ")
print("=====================================================")
print("C = ", param_optimised[0], "+-",np.sqrt(param_covariance_matrix[0,0]))
print("X_mean =", param_optimised[1], "+-",np.sqrt(param_covariance_matrix[1,1]))
print("sigma = ", param_optimised[2], "+-",np.sqrt(param_covariance_matrix[2,2]))
print("actual mean = ", np.mean(x_data))
print("\n")


#STEP 4: PLOTTING THE GAUSSIAN CURVE -----------------------------------------
fig = plt.figure()
x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r.:',label='Gaussian fit')
plt.legend()

#Normalise the histogram values
weights = np.ones_like(x_data) / len(x_data)
plt.hist(x_data, weights=weights)

#setting the label,title and grid of the plot
plt.xlabel("Data: Random variable")
plt.ylabel("Probability")
plt.grid("on")
plt.show()
np.save("param_opt_all_100bins", param_opt_all)
np.save("cov_matrix_all_100bins", cov_matrix_all)