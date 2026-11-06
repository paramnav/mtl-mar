"""
Calculate sediment thickness rate from crustal age and sediment thickness data.

This script computes the ratio of sediment thickness to crustal age,
with interpolation for regions where crustal age is too young.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import sys
from scipy.interpolate import griddata

# Get project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_RAW

# Load datasets
ds_age = xr.open_dataset(
    DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / 
    "SC_CRUST_AGE_MA_Muller.5m.nc"
)
ds_thick = xr.open_dataset(
    DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / 
    "GL_TOT_SED_THICK_M_CRUST1_NOAA.5m.grd"
)

# Extract the actual data arrays
age = np.array(ds_age['data'].squeeze())
thickness = np.array(ds_thick['data'].squeeze())

# Set threshold for minimum crustal age (in million years)
age_threshold = 1  # Adjust if needed

# Compute ratio safely (avoid division by zero for young crust)
safe_ratio = np.where(age > age_threshold, thickness / age, np.nan)

# Create a mask for valid values (where age > threshold)
valid_mask = age > age_threshold

# Get indices of valid and invalid points
valid_indices = np.where(valid_mask)
invalid_indices = np.where(~valid_mask)

# Calculate ratio for valid points
ratio = np.zeros_like(age, dtype=float)
ratio[valid_mask] = thickness[valid_mask] / age[valid_mask]

# Interpolate invalid points using valid neighbors
coords_valid = np.column_stack((valid_indices[0], valid_indices[1]))
coords_invalid = np.column_stack((invalid_indices[0], invalid_indices[1]))

# Interpolate invalid points
ratio[~valid_mask] = griddata(
    coords_valid, ratio[valid_mask], coords_invalid, 
    method='linear', fill_value=np.nan
)

# For any remaining NaN values after interpolation, use nearest neighbor
if np.any(np.isnan(ratio)):
    nan_mask = np.isnan(ratio)
    ratio[nan_mask] = griddata(
        coords_valid, ratio[valid_mask], 
        np.column_stack(np.where(nan_mask)), 
        method='nearest', fill_value=np.nan
    )

# Create new dataset with the interpolated ratio data
lat = np.linspace(-90, 90, 2160)
lon = np.linspace(-180, 180, 4320)

ds_ratio = xr.Dataset(
    data_vars={
        'data': (['lat', 'lon'], ratio),
    },
    coords={
        'lat': lat,
        'lon': lon
    }
)

# Add attributes
ds_ratio['data'].attrs['units'] = 'm/Ma'
ds_ratio['data'].attrs['long_name'] = 'Sediment Thickness per Age'
ds_ratio['data'].attrs['description'] = (
    'Ratio of sediment thickness to crustal age with interpolation for young crust'
)

# Save to netCDF file
output_path = (
    DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / 
    "GL_SC_THICKNESS_PER_AGE.nc"
)
ds_ratio.to_netcdf(output_path)
print(f"Saved sediment thickness rate to: {output_path}")

# Plot the interpolated ratio
plt.figure(figsize=[20, 10], dpi=100)
plt.imshow(ratio, origin='lower', cmap='viridis')
plt.colorbar(label='Sediment Thickness / Crust Age (m/Ma)')
plt.title(f'Interpolated Ratio Map (Age ≤ {age_threshold} Ma interpolated)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()
