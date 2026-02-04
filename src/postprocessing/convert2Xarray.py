import numpy as np
import xarray as xr

def convert_to_xarray(data,  data_name='data'):
    """
    Convert a 2D NumPy array to an xarray.DataArray.

    Parameters:
        data (numpy.ndarray): 2D array containing the data to be converted.
        lats (numpy.ndarray): 1D array containing the latitude values.
        lons (numpy.ndarray): 1D array containing the longitude values.
        data_name (str): Name of the data variable.

    Returns:
        xarray.DataArray: The converted DataArray.
    """
    lats = np.linspace(-90, 90, data.shape[0])
    lons = np.linspace(-180, 180, data.shape[1])

    data_array = xr.DataArray(
        data,
        dims=['latitude', 'longitude'],
        coords={'latitude': lats, 'longitude': lons},
        name=data_name
    )
        
    return data_array

