#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 07:27:22 2025

@author: nparameswaran
"""

import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt
import matplotlib.pyplot as plt

def radial_mean_filter(data, radius_km, lat_res_deg=0.0833, lon_res_deg=0.0833):
    # Estimate number of pixels for 5 km radius
    km_per_deg_lat = 111.32
    pixel_radius = int(np.ceil(radius_km / (lat_res_deg * km_per_deg_lat)))

    # Create a circular kernel mask
    y, x = np.ogrid[-pixel_radius:pixel_radius+1, -pixel_radius:pixel_radius+1]
    mask = x**2 + y**2 <= pixel_radius**2
    mask = mask.astype(float)

    # Normalize mask
    mask /= mask.sum()

    # Convolve the data with the mask using FFT or direct convolution
    from scipy.signal import convolve2d
    filtered = convolve2d(data, mask, mode='same', boundary='symm')

    return filtered

# Load the original dataset
plt.figure()
ds = xr.open_dataset("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/GL_SC_THICKNESS_PER_AGE.nc")
# Rename the variable from "thickness_per_age" to "data" for saving
data = ds["data"].values

# Apply radial averaging
filtered_data = radial_mean_filter(data, radius_km=50)

new_da = xr.DataArray(
    filtered_data,
    dims=ds["data"].dims,
    coords=ds["data"].coords,
    name="data"
)

# Save as new dataset
plt.figure()
new_ds = new_da.to_dataset()
new_ds["data"].plot()
new_ds.to_netcdf("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/GL_SC_THICKNESS_PER_AGE_r50km.nc")