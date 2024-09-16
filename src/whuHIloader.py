import os
import re
import numpy as np
import scipy.io


def whuHi_load(hsi_directory, ground_truth_path, var_header=''):
    """
    Loads hyperspectral image (HSI) data from .mat files and corresponding ground truth data.

    Parameters:
    - hsi_directory: str, path to the directory containing the .mat files with HSI data.
    - ground_truth_path: str, path to the .mat file containing ground truth data.

    Returns:
    - hsi_cube: np.ndarray, a 3D array where the third dimension represents the spectral bands.
    - ground_truth: np.ndarray or None, the ground truth data array or None if not found.
    """

    # List all .mat files in the HSI directory
    file_list = [f for f in os.listdir(hsi_directory) if f.endswith('.mat')]

    # Extract the numeric part (bandwidth or index) and sort the file list accordingly
    sorted_file_list = sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Initialize a list to store the 2D arrays for the HSI cube
    images = []

    # Iterate over the sorted file list
    for filename in sorted_file_list:
        if var_header == '':
            # Construct the key from the filename (without the .mat extension)
            key = os.path.splitext(filename)[0]
        else:
            key = var_header + 't' + os.path.splitext(filename)[0][1:]

        # Load the .mat file
        mat_contents = scipy.io.loadmat(os.path.join(hsi_directory, filename))

        # Access the data using the key
        if key in mat_contents:
            image = mat_contents[key]
            # Append the image to the list
            images.append(image)
        else:
            print(f"Key '{key}' not found in {filename}")

    # Stack the images into a 3D array (HSI cube)
    hsi_cube = np.stack(images, axis=-1)

    # Extract the ground truth key from the file name (without .mat extension)
    ground_truth_key = os.path.splitext(os.path.basename(ground_truth_path))[0]

    # Load the ground truth data
    ground_truth_data = scipy.io.loadmat(ground_truth_path)

    # Access the ground truth data using the key
    if ground_truth_key in ground_truth_data:
        ground_truth = ground_truth_data[ground_truth_key]
    else:
        print(f"Ground truth key '{ground_truth_key}' not found in {ground_truth_path}")
        ground_truth = None

    return hsi_cube, ground_truth

# Example usage:
# hsi_cube, ground_truth = whuHi_load('/path/to/hsi/files', '/path/to/ground_truth.mat')