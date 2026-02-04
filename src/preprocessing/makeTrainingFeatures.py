import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import xarray as xr
from pathlib import Path
from shapely.geometry import Point
from os import walk

def find_nearest(value, vector):
    """
    Find the index of the nearest value in a vector to a given value.

    Parameters:
    value (float): The target value.
    vector (array-like): The array containing values.

    Returns:
    int: Index of the nearest value in the vector.
    """
    vector = np.float64(vector) - np.float64(value)
    return np.argmin(np.abs(vector))

def plot_toc_on_map(df, world, cmap='viridis'):
    """
    Plot TOC labels on a world map.

    Parameters:
    df (DataFrame): DataFrame containing latitude, longitude, and TOC columns.
    world (GeoDataFrame): GeoDataFrame of world map.
    cmap (str, optional): Colormap name. Defaults to 'viridis'.
    """
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)  

    gdf.plot(ax=world.plot(figsize=(20, 12)),
             column=df["TOC [%]"],
             cmap=cmap,
             vmin=0,
             vmax=5,
             markersize=4,
             legend=True,
             legend_kwds={"label": "TOC [%]"}
            )

def select_features(features_path, feature_selection_path):
    """
    Select features based on a list of selected feature names.

    Parameters:
    features_path (Path): Path to directory containing feature files.
    feature_selection_path (Path): Path to file containing selected feature names.

    Returns:
    list: List of selected feature files.
    """
    files = [file for (_, _, files) in walk(features_path) for file in files]
    feature_selection = pd.read_csv(feature_selection_path).values
    files.sort()
    print("Actual number of features")
    print(len(files))
    selected_files = [file for file in files if file in feature_selection]
    print("Selected number of features")
    print(len(selected_files))
    return selected_files
    
def process_features(lee_features_path, lee_files, df_clean):
    """
    Process feature files and extract features.

    Parameters:
    features_path (Path): Path to directory containing feature files.
    files (list): List of feature file names.
    df_clean (DataFrame): DataFrame containing cleaned data.

    Returns:
    tuple: Tuple containing features array, X_mean array, and X_std array.
    """
    rootgrp = xr.open_dataset(Path(lee_features_path / lee_files[0]), format = 'NETCDF4')
    latitudes = rootgrp.variables['lat']
    longitudes = rootgrp.variables['lon']
    lee_features = np.empty([len(df_clean),len(lee_files[0:len(lee_files)])])
    lee_features[:] = np.nan
    lee_X_mean = np.empty(0)
    lee_X_std = np.empty(0)

    file_idx = -1
    corrupted_files = 0


    for file in lee_files[0:len(lee_files)]:
        file_idx += 1

        try: 
            rootgrp = xr.open_dataset(Path(lee_features_path / file), format = 'NETCDF4')
        except:
            lee_X_mean = np.append(lee_X_mean,np.nan)
            lee_X_std = np.append(lee_X_std,np.nan)
            corrupted_files=+1
            continue

        dataset = rootgrp.variables['data']
        lee_X_mean = np.append(lee_X_mean,np.nanmean(dataset))
        lee_X_std = np.append(lee_X_std,np.nanstd(dataset))


        for data_idx in range(0,len(df_clean)):
            lat_idx = find_nearest(df_clean.iloc[data_idx,0], latitudes)
            lon_idx = find_nearest(df_clean.iloc[data_idx,1], longitudes)
            lee_features[data_idx, file_idx] = dataset[lat_idx, lon_idx]

        print("file " + str(file_idx) + " :" + str(file)+  " done!")

    print("corrupted files: " + str(corrupted_files))
    temp_features = lee_features[:,:file_idx+1]
    lee_features = temp_features
    features = np.concatenate([lee_features], axis = 1)
    X_mean = np.concatenate([lee_X_mean])
    X_std = np.concatenate([lee_X_std])
    latitudes_labels = np.array(df_clean["Latitude"])
    longitudes_labels = np.array(df_clean["Longitude"])
    
    return lee_features, X_mean, X_std, latitudes_labels, longitudes_labels

def save_data(features, labels, mean, std, lat_labels, lon_labels, save_path):
    """
    Save processed data.

    Parameters:
    features (ndarray): Array of features.
    labels (ndarray): Array of labels.
    mean (ndarray): Array of feature means.
    std (ndarray): Array of feature standard deviations.
    save_path (Path): Path to save the processed data.
    """
    np.save(save_path / "numpy_features", features)
    np.save(save_path / "numpy_labels", labels)
    np.save(save_path / "features_mean", mean)
    np.save(save_path / "features_std", std)
    np.save(save_path / "numpy_lat", lat_labels)
    np.save(save_path / "numpy_lon", lon_labels)

