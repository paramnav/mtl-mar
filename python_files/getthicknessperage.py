import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Get project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_RAW

# Load datasets
ds_age = xr.open_dataset(DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / "SC_CRUST_AGE_MA_Muller.5m.nc")
ds_thick = xr.open_dataset(DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / "GL_TOT_SED_THICK_M_CRUST1_NOAA.5m.grd")

# Extract the actual data arrays
age = ds_age['data'].squeeze()
thickness = ds_thick['data'].squeeze()

age = np.array(age)
thickness = np.array(thickness)

plt.figure(figsize=[20, 10], dpi =100)
plt.imshow(thickness/age, vmax = 100)
plt.colorbar()

import numpy as np
import matplotlib.pyplot as plt

# Convert to NumPy arrays (already done)
age = np.array(age)
thickness = np.array(thickness)


# Set threshold
age_threshold = 1  # you can adjust this

# Compute ratio safely
safe_ratio = np.where(age > age_threshold, thickness / age, np.nan)

# Create a mask: 1 where age is too small, else 0
mask = (age <= age_threshold).astype(float)
"""
# Plot main ratio map
plt.figure(figsize=[20, 10], dpi=100)
plt.imshow(safe_ratio, origin='lower', cmap='viridis', vmax=100)
plt.colorbar(label='Sediment Thickness / Crust Age')
plt.title(f'Ratio Map (Red = Age ≤ {age_threshold})')
"""
# Create a mask for valid values (where age > threshold)
valid_mask = age > age_threshold

# Get indices of valid and invalid points
valid_indices = np.where(valid_mask)
invalid_indices = np.where(~valid_mask)

# Calculate ratio for valid points
ratio = np.zeros_like(age, dtype=float)
ratio[valid_mask] = thickness[valid_mask] / age[valid_mask]

# Interpolate invalid points using valid neighbors
from scipy.interpolate import griddata

# Get coordinates of valid and invalid points
coords_valid = np.column_stack((valid_indices[0], valid_indices[1]))
coords_invalid = np.column_stack((invalid_indices[0], invalid_indices[1]))

# Interpolate invalid points
ratio[~valid_mask] = griddata(coords_valid, ratio[valid_mask], coords_invalid, method='linear', fill_value=np.nan)

# For any remaining NaN values after interpolation, use nearest neighbor
if np.any(np.isnan(ratio)):
    nan_mask = np.isnan(ratio)
    ratio[nan_mask] = griddata(coords_valid, ratio[valid_mask], 
                              np.column_stack(np.where(nan_mask)), 
                              method='nearest', fill_value=np.nan)
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
ds_ratio['data'].attrs['description'] = 'Ratio of sediment thickness to crustal age with interpolation for young crust'

# Save to netCDF file
ds_ratio.to_netcdf(DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated" / "GL_SC_THICKNESS_PER_AGE.nc")

# Plot the interpolated ratio
plt.figure(figsize=[20, 10], dpi=100)
plt.imshow(ratio, origin='lower', cmap='viridis')
plt.colorbar(label='Sediment Thickness / Crust Age (Interpolated)')
plt.title(f'Interpolated Ratio Map (Age ≤ {age_threshold} interpolated)')

