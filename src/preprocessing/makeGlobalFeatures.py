    
import numpy as np
import pandas as pd
import xarray as xr
from os import walk
from pathlib import Path
from multiprocessing import Pool



def get_restrepo_files(restrepo_input_path):
    """
    Get a list of files in the Restrepo feature directory.

    Parameters:
    restrepo_input_path (Path): Path to the Restrepo feature directory.

    Returns:
    list: List of file names.
    """
    restrepo_files = []
    for (dirpath, dirnames, filenames) in walk(restrepo_input_path):
        restrepo_files.extend(filenames)
        break
    restrepo_files.sort()
    return restrepo_files

def load_datasets(restrepo_files, restrepo_input_path, feature_selection):
    """
    Load datasets from Restrepo feature files.

    Parameters:
    restrepo_files (list): List of file names.
    restrepo_input_path (Path): Path to the Restrepo feature directory.

    Returns:
    list: List of loaded datasets.
    int: Number of corrupted files.
    """
    
    corrupted_files = 0
    for file in restrepo_files:
        if file in feature_selection:
            try:
                rootgrp = xr.open_dataset(Path(restrepo_input_path / file), format='NETCDF4')
                dataset = rootgrp.variables['data']
                datasets.append(dataset)
            except:
                datasets.append(np.nan)
                corrupted_files += 1
    return dataset, datasets, corrupted_files



