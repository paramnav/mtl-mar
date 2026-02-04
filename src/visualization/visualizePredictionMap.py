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

from pathlib import Path
from matplotlib.colors import LogNorm
################################################################################
#"""
filename = "/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/prediction_map_TOC_noconstraint_supervised_men_CS_CNN.npy"
prediction_map_CS = np.load(filename)
prediction_map_DO = np.load("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/predictionmaps_TOC/prediction_map_TOC_noconstraint_supervised_men_DO_CNN.npy")


dataset_mask = np.load("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/masks/mask_deepocean.npy")
dataset_mask = np.rot90(np.rot90(np.fliplr(dataset_mask)))

# Create a copy of pm_CS
pm_overlapped = np.copy(prediction_map_DO)

# Mask the values from pm_CS where mask contains NaN values
pm_overlapped[np.isnan(dataset_mask)] = prediction_map_CS[np.isnan(dataset_mask)]

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
"""
plt.show()

ds=2.6

file_porosity = "/home/sunms498/Data/porosity/grl53425-sup-0002-supinfo.grd"
#file_porosity = "file_porosity = "/home/sunms498/Data/paper_porosity/GPSM_Restreppo/prediction.nc""
rootgrp_porosity = xr.open_dataset(Path(file_porosity), format = 'NETCDF4')
                                                                   




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


mar_naveen_nonan = pm_overlapped
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Generate some example data for demonstration

plt.figure(figsize=(20, 10), dpi=300)
plt.imshow(mar_naveen_nonan, norm=LogNorm(vmin=1e-1, vmax=10), cmap="cubehelix", extent=[-180, 180, -90, 90])
#plt.title(filename)
# Commenting out the title
# plt.title("Mass Accumulation Rate Using K-Nearest Neighbours", fontsize=14, fontweight='bold')

# Add text at the bottom of the plot
#additional_text = "k = 8  |  Sediment Budget: $6.4 \\times 10^4$ Mt/yr"
#plt.text(0.5, -0.08, additional_text, transform=plt.gca().transAxes, fontsize=10, ha='center', color='gray')
# Adjust the size of the color bar using the shrink parameter
cbar = plt.colorbar(label='TOC [%]', aspect=40, pad=0.015, shrink=0.8)
cbar.ax.tick_params(labelsize=20)  # Increase font size for color bar ticks and labels
cbar.set_label('TOC [%]', fontsize=20)
plt.title("TOC[%]: Conv Neural Net", fontsize=14, fontweight='bold')


plt.xticks(fontsize=20)  # Increase font size for x-axis ticks
plt.yticks(fontsize=20)

plt.tight_layout()

# Save the plot
#plt.savefig('mass_accumulation_plot.png', dpi=300)

plt.show()


"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Replace `mar_naveen_nonan` and `north_sea_bounds` with your data and region bounds
# mar_naveen_nonan = ...

# Define the boundaries for the North Sea region (modify as needed)
north_sea_bounds = [-10, 10, 50, 62]

# Create a figure
plt.figure(figsize=(10, 10), dpi=100)

# Plot the data using imshow
plt.imshow(mar_naveen_nonan, norm=LogNorm(vmin=1e-5, vmax=5), cmap="cubehelix", extent=[-180, 180, -90, 90])

# Set the extent to focus on the North Sea region
plt.xlim(north_sea_bounds[0], north_sea_bounds[1])
plt.ylim(north_sea_bounds[2], north_sea_bounds[3])

# Add labels and title (uncomment if needed)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.title('Mass Accumulation Rate in the North Sea', fontsize=20)

# Add a color bar with manually calculated aspect
cax = plt.colorbar(label='Mass Accumulation Rate', pad=0.015, shrink=0.46)
cax.ax.tick_params(labelsize=20)
cax.set_label('Mass Accumulation Rate (g/cm²/yr)', fontsize=20)

# Manually calculate the aspect ratio to match the plot size
aspect = 10  # Adjust this value as needed to match your plot size
cax.ax.set_aspect(aspect)

# Increase font size for ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()

# Show the plot
plt.show()


# Load data from the 6 npy files
mar_constraint01 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint01.npy')
mar_constraint02 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint02.npy')
mar_constraint03 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint03.npy')
mar_constraint04 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint04.npy')
mar_constraint05 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint05.npy')
mar_constraint0 = np.load('/home/sunms498/Downloads/klaus_meeting/mar_constraint0.npy')

# Combine the data into a single array (modify the order as needed)
mar_naveen_nonan = np.stack([mar_constraint0,mar_constraint01, mar_constraint02, mar_constraint03, mar_constraint04, mar_constraint05])


# Define the boundaries for the North Sea region (modify as needed)
north_sea_bounds = [-10, 10, 50, 62]

# Create a 3x2 grid of plots with shared x and y axes
fig, axs = plt.subplots(3, 2, figsize=(20, 20), dpi=100, sharex=True, sharey=True)
constraint_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Plot the data using imshow for each subplot
for i, ax in enumerate(axs.flatten()):
    im = ax.imshow(mar_naveen_nonan[i], norm=LogNorm(vmin=1e-5, vmax=5), cmap="cubehelix", extent=[-180, 180, -90, 90])
    ax.set_xlim(north_sea_bounds[0], north_sea_bounds[1])
    ax.set_ylim(north_sea_bounds[2], north_sea_bounds[3])
    # Annotate the subplot with the constraint value
    constraint_value = constraint_values[i]  # Modify as needed
    ax.annotate(f'$\lambda$: {constraint_value}', xy=(0.85, 0), xycoords='axes fraction',
                fontsize=20, ha='right', va='bottom')
# Manually set labels for the axes if needed
axs[2, 0].set_xlabel('Longitude', fontsize=20, labelpad=10)  
axs[2, 0].xaxis.set_label_coords(1.05, -0.1)  
axs[1, 0].set_ylabel('Latitude', fontsize=20)

# Add a color bar centered on the last subplot
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position and size to center the color bar
cbar = plt.colorbar(im, cax=cax, label='Mass Accumulation Rate')
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Mass Accumulation Rate (g/cm²/yr)', fontsize=20)

# Increase font size for ticks on all subplots
for ax in axs.flatten():
    ax.tick_params(labelsize=20)

# Remove titles
for ax in axs.flatten():
    ax.set_title('')

# Remove white spaces between the plots
plt.subplots_adjust(wspace=0, hspace=0)

axs[2,0].set_xticks(axs[2,0].get_xticks()[:-1])

# Remove the leftmost x-tick in the left column

axs[0,1].set_xticks(axs[0,1].get_xticks()[1:])
"""
# Show the plot
plt.show()
# Save the plot with high resolution for publication
#plt.savefig('published_plot.png', dpi=300, bbox_inches='tight')
"""
plt.show()
plt.figure(figsize=([20,10]),dpi=300)
plt.imshow(area)
plt.title("Area of grid cell")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
plt.axis("off")
plt.colorbar(label='Area(km2) ')
#plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
"""


#mar_restreppo = 10**dataset_numpy_2021

# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_restreppo,norm=LogNorm(vmin=1e-5, vmax=5),cmap="cubehelix")
# plt.title("Global mass accumulation rate from R21 paper")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$]')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")

print(int(np.sum(area * mar_naveen_nonan)))

# prediction_map_shallow = np.load("/home/sunms498/work/toc-neural-nets/Sed_regresion_shallow_RestrepoFeats_AllVS0.2_noNaN.npy")
# prediction_map_shallow_flipped= np.rot90(np.rot90(np.fliplr(prediction_map_shallow)))
# prediction_map_shallow_flipped[land_map]=np.nan


# mar_naveen_shallow_nonan = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_shallow_flipped/1e3)
# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen_shallow_nonan,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction(trained with only sed rates from shallow regions(>-200m)) from neural network model(ds=2.6)(noNan values)")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log

# prediction_map_deep = np.load("/home/sunms498/work/toc-neural-nets/Sed_regresion_deep_RestrepoFeats_AllVS0.2_noNaN.npy")
# prediction_map_deep_flipped= np.rot90(np.rot90(np.fliplr(prediction_map_deep)))
# prediction_map_deep_flipped[land_map]=np.nan


# mar_naveen_deep_nonan = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_deep_flipped/1e3)
# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen_deep_nonan,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction(trained with only sed rates from deep regions(<-200m)) from neural network model(ds=2.6)(noNan values)")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log

# mar_combined_model = np.zeros([prediction_map_deep_flipped.shape[0], prediction_map_deep_flipped.shape[1]])
# mask_shallow = np.load("mask_shallow.npy")
# mask_shallow = np.rot90(np.rot90(np.fliplr(mask_shallow)))
# for i in range(mask_shallow.shape[0]):
#     for j in range(mask_shallow.shape[1]):
#         if mask_shallow[i][j]==255:
#             mar_combined_model[i][j]=mar_naveen_shallow_nonan[i][j]
#         else:
#             mar_combined_model[i][j]=mar_naveen_deep_nonan[i][j]

# x_labels=[-180, -90, 0,  90, 180 ]#np.linspace(-180,180,4320)
# fig, ax = plt.subplots(figsize=([20,10]),dpi=300)
# ax.set_xticklabels(x_labels)
# plt.imshow(mar_combined_model,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction, combining deep and shallow sed rates from neural network model(ds=2.6)(noNan values)")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')

###############################################################################

# prediction_map_nolimit = np.load("/home/sunms498/Data/prediction_map_noNAN_nolimit_1000x6.npy")
# prediction_map_nolimit = 10**prediction_map_nolimit -1
# prediction_map_nolimit_flipped= np.rot90(np.rot90(np.fliplr(prediction_map_nolimit)))
# prediction_map_nolimit_flipped[land_map]=np.nan




# mar_naveen_nolimit_nonan = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_nolimit_flipped/1e3)
# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen_nolimit_nonan,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction trained with all the data: regression")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log
            

# variance_map_nolimit = np.load("/home/sunms498/Data/variance_map_noNAN_nolimit_1000x6.npy")
# variance_map_nolimit_flipped= np.rot90(np.rot90(np.fliplr(variance_map_nolimit)))
# variance_map_nolimit_flipped[land_map]=np.nan


#

# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(variance_map_nolimit_flipped,norm=LogNorm(vmin=1e-3, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate variance trained with all the data: regression")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log

###############################################################################
# prediction_map_nolimit_bayes = np.load("/home/sunms498/Data/prediction_map_noNAN_nolimit_bayes.npy")
# prediction_map_nolimit_bayes = 10**prediction_map_nolimit_bayes -1
# prediction_map_nolimit_bayes_flipped= np.rot90(np.rot90(np.fliplr(prediction_map_nolimit_bayes)))
# #prediction_map_nolimit_bayes_flipped[land_map]=np.nan



# mar_naveen_nolimit_nonan_bayes = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_nolimit_bayes_flipped/1e3)
# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen_nolimit_nonan_bayes,norm=LogNorm(vmin=1e-3, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction trained with all the data: bayes")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log
            

# variance_map_nolimit_bayes = np.load("/home/sunms498/Data/variance_map_noNAN_nolimit_bayes.npy")
# variance_map_nolimit_bayes_flipped= np.rot90(np.rot90(np.fliplr(variance_map_nolimit_bayes)))
# variance_map_nolimit_bayes_flipped[land_map]=np.nan




# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(variance_map_nolimit_bayes_flipped,cmap="turbo")
# plt.title("Mass accumulation rate variance trained with all the data: bayes")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log
###############################################################################

# prediction_map_nolimit_softmax = np.load("/home/sunms498/Data/prediction_map_noNAN_nolimit_softmax.npy")
# #prediction_map_nolimit_softmax = 10**prediction_map_nolimit_softmax -1
# prediction_map_nolimit_softmax_flipped= np.rot90(np.rot90(np.fliplr(prediction_map_nolimit_softmax)))
# #prediction_map_nolimit_softmax_flipped[land_map]=np.nan



# mar_naveen_nolimit_nonan_softmax = (1-(dataset_numpy_porosity/100))*ds*(prediction_map_nolimit_softmax_flipped/1e3)
# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(mar_naveen_nolimit_nonan_softmax,norm=LogNorm(vmin=1e-5, vmax=5),cmap="turbo")
# plt.title("Mass accumulation rate prediction trained with all the data: softmax")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log
            

# variance_map_nolimit_softmax = np.load("/home/sunms498/Data/variance_map_noNAN_nolimit_softmax.npy")
# variance_map_nolimit_softmax_flipped= np.rot90(np.rot90(np.fliplr(variance_map_nolimit_softmax)))
# variance_map_nolimit_softmax_flipped[land_map]=np.nan




# plt.figure(figsize=([20,10]),dpi=300)
# plt.imshow(variance_map_nolimit_softmax_flipped,cmap="turbo")
# plt.title("Mass accumulation rate variance trained with all the data: softmax")#(ours)(converted with MAR=(1-p)(ds)(w); ds $= 2.6$, w: sedimentation rate and p: porosity)
# plt.axis("off")
# plt.colorbar(label='Mass accumulation rate[$g/cm^2/yr$] ')
# #plt.savefig( "/home/sunms498/Data/MassAccumulationRateNaveen"+".png")
# #plt.clim(0, 30)
# #wihtout log
