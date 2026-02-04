import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Colormap

def VisualiseMap(data, cmap, use_log_norm=False, vmin=None, vmax=None, cbar_label=None, 
                 lat_range=None, lon_range=None):
    """
    Visualize a map using matplotlib.

    Parameters:
        data (numpy.ndarray): Array containing the data to be visualized.
        cmap (str or matplotlib.colors.Colormap): Colormap to be used for visualization.
        use_log_norm (bool): Whether to use logarithmic normalization for color scaling.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
        cbar_label (str): Label for the colorbar.
        lat_range (tuple): Tuple specifying the latitude range (min_lat, max_lat).
        lon_range (tuple): Tuple specifying the longitude range (min_lon, max_lon).
    """
    plt.figure(figsize=(20, 10), dpi=300)
    

    
    if lat_range and lon_range:
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range
        
        # Calculate the indices corresponding to the specified latitude and longitude ranges
        lat_indices = np.linspace(90, -90, data.shape[0])
        lon_indices = np.linspace(-180, 180, data.shape[1])
        print(lat_indices)
        lat_mask = (lat_indices >= min_lat) & (lat_indices <= max_lat)
        lon_mask = (lon_indices >= min_lon) & (lon_indices <= max_lon)
        
        # Ensure that the mask applies correctly to the data
        lat_indices = np.linspace(-90, 90, data.shape[0])
        lon_indices = np.linspace(-180, 180, data.shape[1])
        
        lat_mask = (lat_indices >= min_lat) & (lat_indices <= max_lat)
        lon_mask = (lon_indices >= min_lon) & (lon_indices <= max_lon)
        
        data_snippet = data[np.ix_(lat_mask, lon_mask)]
        extent = [min_lon, max_lon, min_lat, max_lat]
    else:
        data_snippet = data
        extent = [-180, 180, -90, 90]

    if use_log_norm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plt.imshow(data_snippet, cmap=cmap, norm=norm, extent=extent)
        
    else:
        norm = None
        
    if use_log_norm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plt.imshow(data_snippet, cmap=cmap, norm=norm,  extent=extent)
    else:
        norm = None
        plt.imshow(data_snippet, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, extent=extent)
    
    cbar = plt.colorbar(label=cbar_label, aspect=40, pad=0.01, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, fontsize=20)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Longitude", fontsize=20)
    plt.ylabel("Latitude", fontsize=20)
    plt.tight_layout()
    
    plt.show()
        
