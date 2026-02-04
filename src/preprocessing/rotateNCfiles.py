
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from os import walk
import pickle
from pathlib import Path
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point

def find_nearest(value,vector):
    vector = np.float64(vector)-np.float64(value);
    return np.argmin(np.abs(vector))
#restrepo_features_path = Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/")
#restrepo_features_path = Path("/gxfs_work/cau/sunms498/work_old/Data/Features_PhrampusLee_noNaN_highres_rot/")
restrepo_features_path = Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/")
restrepo_files = []

for (dirpath, dirnames, filenames) in walk(restrepo_features_path):
    #print(filenames)
    restrepo_files.extend(filenames)

restrepo_files.sort()
len(restrepo_files)


feature_selection = pd.read_csv(Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/selectedFeatureLists/selectedfeatures_v_men.txt")).values
#feature_selection = pd.read_csv(Path("restrepoHighResFeaturesRaw-50.txt")).values


selection_filter = np.zeros(len(restrepo_files), dtype=bool)
for idx, feature_name in enumerate(restrepo_files):
    if feature_name in feature_selection: # drop '.nc' from feature names #drop .grd
        selection_filter[idx] = True
feature_selection.shape
filenames
selected_files = []
for file in restrepo_files:
    if file in feature_selection:
        selected_files.append(file)
len(feature_selection)
restrepo_files = selected_files
restrepo_files
file_idx = -1
corrupted_files = 0

for file in restrepo_files[0:len(restrepo_files)]:
    file_idx += 1
    
    try:
        rootgrp = xr.open_dataset(Path(restrepo_features_path / file), format = 'NETCDF4')
        
        if file == "SF_POROSITY_FRC.5m_with_coords_flip.nc":
            porosity_fileidx = file_idx
            print("porosity_fileidx_lee")
            print(porosity_fileidx)
            dataset_numpy_porosity = rootgrp.variables['data']

        elif file == "POROSITY_global_prediction.grd":
            porosity_fileidx_martin = file_idx
            print("porosity_fileidx_martin") 
            print(porosity_fileidx_martin)
            dataset_numpy_porosity_martin = rootgrp.variables['data']

            
        elif file == "SEAFLOOR_DENSITY_GCM3.nc":
            density_fileidx = file_idx
            print("density_fileidx")
            print(density_fileidx)
            dataset_numpy_density = rootgrp.variables['data']
            

    except:
        restrepo_X_mean = np.append(restrepo_X_mean,np.nan)
        restrepo_X_std = np.append(restrepo_X_std,np.nan)
        corrupted_files += 1
        continue