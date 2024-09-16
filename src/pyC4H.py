import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import scipy.io
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import cv2

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors

np.random.seed(0)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected, spectral embedding may not work as expected.")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

# Define a longer list of distinct colors
colors = [
    'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'grey', 'cyan',
    'magenta', 'lime', 'lavender', 'teal', 'olive', 'maroon', 'navy', 'gold', 'coral', 'turquoise',
    'indigo', 'violet', 'silver', 'darkgreen', 'darkblue', 'darkred', 'salmon', 'sienna', 'khaki', 'orchid',
    'beige', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgrey', 'tan', 'plum', 'peru',
    'wheat', 'seagreen', 'slateblue', 'crimson', 'deepskyblue', 'dodgerblue', 'forestgreen', 'fuchsia', 'goldenrod',
    'hotpink',
    'indianred', 'lightseagreen', 'limegreen', 'mediumorchid', 'mediumslateblue', 'mediumturquoise', 'mediumvioletred',
    'midnightblue', 'mistyrose', 'orangered'
]
# Create a ListedColormap using these colors
cmap = mcolors.ListedColormap(colors)


def remove_trailing_zero_rows(matrix):
    # Iterate from the last row to the first
    for i in range(len(matrix) - 1, -1, -1):
        # Check if the current row is all zeros
        if np.any(matrix[i] != 0):
            # If a non-zero row is found, stop and slice the matrix up to this row (inclusive)
            return matrix[:i + 1]
    # If all rows are zero, return an empty matrix with the same number of columns
    return np.empty((0, matrix.shape[1]), dtype=matrix.dtype)


def load_hsi_data(data_path, var_name):
    mat_data = scipy.io.loadmat(data_path)
    hsi_data = mat_data[var_name]
    return hsi_data


def preprocess_hsi_data(hsi_data, kernel_size=None):
    """
    Preprocess (Gaussian denoise) the HSI data.

    Parameters:
    hsi_data (ndarray): The input HSI data with shape (spectral, height, width).
    kernel_size (tuple or None): The kernel size for Gaussian filtering. It can be:
        - None: Return the undenoised HSI data.
        - 2D tuple: Apply a 2D Gaussian filter on each spectral layer.
        - 3D tuple: Apply a 3D Gaussian filter on the entire HSI data.

    Returns:
    denoised_hsi_data (ndarray): The denoised HSI data.
    """
    if kernel_size is None:
        return hsi_data
    elif len(kernel_size) == 2:
        denoised_hsi_data = np.zeros_like(hsi_data)
        for i in range(hsi_data.shape[2]):
            denoised_hsi_data[:, :, i] = gaussian_filter(hsi_data[:, :, i], sigma=kernel_size)
    elif len(kernel_size) == 3:
        denoised_hsi_data = gaussian_filter(hsi_data, sigma=kernel_size)
    else:
        raise ValueError("Kernel size must be None, a 2D tuple, or a 3D tuple.")

    return denoised_hsi_data


def apply_pca(reshaped_data, n_components):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reshaped_data)

    if n_components is None or n_components <= 0:
        return scaled_data

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data


def cluster_gmm(data, n_components, covariance_type, init_params, max_iter, random_state):
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,
                          init_params=init_params,
                          max_iter=max_iter,
                          random_state=random_state)
    labels = gmm.fit_predict(data)
    return labels


def cluster_dbscan(data, epsilon, minPts):
    db = DBSCAN(eps=epsilon, min_samples=minPts, n_jobs=-1).fit(data)
    labels = db.labels_
    return labels


def cluster_hdbscan(data, min_samples, min_cluster_size):
    hdbscan = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
    labels = hdbscan.fit_predict(data)
    return labels


def cluster_spectral(data, n_clusters, n_components, n_neighbors, assign_labels, affinity, gamma=None):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, gamma=gamma, random_state=0, n_jobs=-1,
                                  eigen_solver='amg', n_components=n_components, n_neighbors=n_neighbors,
                                  assign_labels=assign_labels)
    labels = spectral.fit_predict(data)
    return labels


def cluster_kmeans(data, n_clusters, init, n_init, max_iter, tol, random_state):
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                    random_state=random_state).fit(data)
    labels = kmeans.labels_
    return labels


# TODO: Now sorting matching => TBD Hungarian Optimization (O(n^3), dummy rows for non square complicate the prob,
#  challenge to implement for sparse)
def match_labels(gt, labels):
    # 1: All inside the gt mask
    # 2: Partial inside the mask
    # 3: TOTOAL Out of the mask

    # Task: Optimize matching in mask to benefit metrics
    # STEP1: (1, 2) match labels from gt_labels (NEVER 0) to assign them, TODO: left r2i (NOT IMP)
    # TODO: STEP1.1: (2) ASSIGN 0 to 2i iff #r2i > max(#3i) (FOR GOOD LOOKING)
    # STEP2: (3) ASSIGN 0 to max_i*(#3i*) for good looking iff ~STEP1.1
    # STEP3: (3_exclude3i*) lift them up to avoid duplicating

    # Initialize label mapping dictionary
    label_map = {}
    used_labels = set()  # Track used predicted labels to ensure uniqueness
    mask12 = gt > 0

    # Get unique ground truth labels
    gts = gt[mask12]
    lbs = labels[mask12]
    u_gts, c_gts = np.unique(gts, return_counts=True)

    # STEP 1: Match labels from `gt` inside the mask
    for u_gt_i, c_gt_i in sorted(zip(u_gts, c_gts), key=lambda x: -x[1]):
        # Find predicted labels corresponding to the current ground truth label
        lbs_4_gt_i = lbs[gts == u_gt_i]

        # Count occurrences of each predicted label
        u_lbs_4_gt_i, c_lbs_4_gt_i = np.unique(lbs_4_gt_i, return_counts=True)

        # Assign the most frequent predicted label to the current ground truth label
        for u_lbs_4_gt_i_j, c_lbs_4_gt_i_j in sorted(zip(u_lbs_4_gt_i, c_lbs_4_gt_i), key=lambda x: -x[1]):
            if u_lbs_4_gt_i_j not in used_labels:
                label_map[u_lbs_4_gt_i_j] = u_gt_i
                used_labels.add(u_lbs_4_gt_i_j)
                break
        # else:
        # If all predicted labels are used TODO: THIS SHOULD NOT MATTER
        # print("Whatever, THIS CAN HAPPEN BUT, SHOULD NOT MATTER. DOES NOT INTRODUCE A ZERO ROW")

    # STEP 2: Handle labels outside the mask, or left in STEP 1
    unique_pred_labels = np.unique(labels)
    unmatched_pred_labels = set(unique_pred_labels) - set(label_map.keys())

    if unmatched_pred_labels:
        for pred_label in unmatched_pred_labels:
            new_label = np.max(list(u_gts)) + 1 + pred_label
            label_map[pred_label] = new_label

    return label_map


def optimize_diagonal_elements(conf_matrix, row_labels, col_labels, thrshold=0.017):
    """
    Optimize the diagonal elements by identifying rows or columns where the diagonal element is zero
    and suggesting merges with the row or column containing the largest off-diagonal element.

    Parameters:
    conf_matrix (ndarray): The confusion matrix to be optimized.
    row_labels (list): The labels corresponding to the rows of the confusion matrix.
    col_labels (list): The labels corresponding to the columns of the confusion matrix.

    Returns:
    reordered_conf_matrix (ndarray): The reordered confusion matrix.
    merge_map (dict): Map indicating the merges with the format (index, 'row'/'col'): merge_with_index.
    reordered_row_labels (list): The reordered labels for rows.
    reordered_col_labels (list): The reordered labels for columns.
    """
    n_rows, n_cols = conf_matrix.shape
    zero_diag_rows = [i for i in range(n_rows) if conf_matrix[i, i] < np.sum(conf_matrix) * thrshold]
    zero_diag_cols = [i for i in range(n_cols) if conf_matrix[i, i] < np.sum(conf_matrix) * thrshold]

    merge_map = {}
    row_order = list(range(n_rows))
    col_order = list(range(n_cols))

    for row in zero_diag_rows:
        if np.sum(conf_matrix[row]) / np.sum(conf_matrix) > thrshold:
            largest_elem_col = np.argmax(conf_matrix[row, :])
            if largest_elem_col != row:
                merge_map[(row_labels[row], 'row')] = row_labels[largest_elem_col]
                row_order.remove(row)
                row_order.insert(row_order.index(largest_elem_col) + 1, row)
    for col in zero_diag_cols:
        if np.sum(conf_matrix[:, col]) / np.sum(conf_matrix) > thrshold:
            largest_elem_row = np.argmax(conf_matrix[:, col])
            if largest_elem_row != col:
                merge_map[(col_labels[col], 'col')] = col_labels[largest_elem_row]
                col_order.remove(col)
                col_order.insert(col_order.index(largest_elem_row) + 1, col)

    reordered_conf_matrix = conf_matrix[np.ix_(row_order, col_order)]
    reordered_row_labels = [row_labels[i] for i in row_order]
    reordered_col_labels = [col_labels[i] for i in col_order]

    return reordered_conf_matrix, merge_map, reordered_row_labels, reordered_col_labels


# Example usage:
# conf_matrix = np.array([[...]])  # Your confusion matrix
# row_labels = ["row1", "row2", ...]  # Your row labels
# col_labels = ["col1", "col2", ...]  # Your column labels
# reordered_conf_matrix, merge_map, reordered_row_labels, reordered_col_labels = optimize_diagonal_elements(conf_matrix, row_labels, col_labels)
# sorted_conf_matrix, sorted_row_labels, sorted_col_labels = sort_confusion_matrix(reordered_conf_matrix, reordered_row_labels, reordered_col_labels, merge_map)


def sort_confusion_matrix(conf_matrix, row_labels, col_labels, merge_map=None):
    """
    Sort the confusion matrix and labels based on the sums of the rows and columns,
    considering merged widths or heights if merge_map is provided.

    Parameters:
    conf_matrix (ndarray): The confusion matrix to be sorted.
    row_labels (list): The labels corresponding to the rows of the confusion matrix.
    col_labels (list): The labels corresponding to the columns of the confusion matrix.
    merge_map (dict): Optional map indicating merges with the format (index, 'row'/'col'): merge_with_index.

    Returns:
    sorted_conf_matrix (ndarray): The sorted confusion matrix.
    sorted_row_labels (list): The sorted labels for rows.
    sorted_col_labels (list): The sorted labels for columns.
    """
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)

    # Create dictionaries to track cumulative sums
    cumulative_row_sums = {label: row_sums[i] for i, label in enumerate(row_labels)}
    cumulative_col_sums = {label: col_sums[i] for i, label in enumerate(col_labels)}

    if merge_map:
        # First iteration: accumulate the sums according to merge_map
        for (index, merge_type), merge_with_index in merge_map.items():
            if merge_type == 'row':
                cumulative_row_sums[merge_with_index] += cumulative_row_sums[index]
            elif merge_type == 'col':
                cumulative_col_sums[merge_with_index] += cumulative_col_sums[index]

        # Second iteration: assign the accumulated sums to all relevant indices
        for (index, merge_type), merge_with_index in merge_map.items():
            if merge_type == 'row':
                cumulative_row_sums[index] = cumulative_row_sums[merge_with_index]
            elif merge_type == 'col':
                cumulative_col_sums[index] = cumulative_col_sums[merge_with_index]

    # Create sorted indices based on cumulative sums
    sorted_row_indices = sorted(range(len(row_labels)), key=lambda i: cumulative_row_sums[row_labels[i]], reverse=True)
    sorted_col_indices = sorted(range(len(col_labels)), key=lambda i: cumulative_col_sums[col_labels[i]], reverse=True)

    sorted_conf_matrix = conf_matrix[sorted_row_indices, :]
    sorted_conf_matrix = sorted_conf_matrix[:, sorted_col_indices]

    sorted_row_labels = [row_labels[i] for i in sorted_row_indices]
    sorted_col_labels = [col_labels[i] for i in sorted_col_indices]

    return sorted_conf_matrix, sorted_row_labels, sorted_col_labels


# Example usage:
# conf_matrix = np.array([[...]])  # Your confusion matrix
# row_labels = ["row1", "row2", ...]  # Your row labels
# col_labels = ["col1", "col2", ...]  # Your column labels
# reordered_conf_matrix, merge_map, reordered_row_labels, reordered_col_labels = optimize_diagonal_elements(conf_matrix, row_labels, col_labels)
# sorted_conf_matrix, sorted_row_labels, sorted_col_labels = sort_confusion_matrix(reordered_conf_matrix, reordered_row_labels, reordered_col_labels, merge_map)


def merge_confusion_matrix(sorted_conf_matrix, sorted_row_labels, sorted_col_labels, merge_map):
    """
    Merge rows and columns in the sorted confusion matrix based on the merge map.

    Parameters:
    sorted_conf_matrix (ndarray): The sorted confusion matrix.
    sorted_row_labels (list): The sorted labels for rows.
    sorted_col_labels (list): The sorted labels for columns.
    merge_map (dict): Map indicating the merges with the format (label, 'row'/'col'): merge_with_label.

    Returns:
    merged_conf_matrix (ndarray): The merged confusion matrix.
    merged_row_labels (list): The merged labels for rows.
    merged_col_labels (list): The merged labels for columns.
    """
    row_label_map = {label: i for i, label in enumerate(sorted_row_labels)}
    col_label_map = {label: i for i, label in enumerate(sorted_col_labels)}

    merged_conf_matrix = sorted_conf_matrix.copy()

    for (label, merge_type), merge_with_label in merge_map.items():
        try:
            if merge_type == 'row':
                row_idx = row_label_map[label]
                merge_with_idx = row_label_map[merge_with_label]
                merged_conf_matrix[merge_with_idx, :] += merged_conf_matrix[row_idx, :]
                merged_conf_matrix = np.delete(merged_conf_matrix, row_idx, axis=0)
                del row_label_map[label]
                # Update row_label_map to account for removed index
                row_label_map = {label: (i if i < row_idx else i - 1) for label, i in row_label_map.items()}
            elif merge_type == 'col':
                col_idx = col_label_map[label]
                merge_with_idx = col_label_map[merge_with_label]
                merged_conf_matrix[:, merge_with_idx] += merged_conf_matrix[:, col_idx]
                merged_conf_matrix = np.delete(merged_conf_matrix, col_idx, axis=1)
                del col_label_map[label]
                # Update col_label_map to account for removed index
                col_label_map = {label: (i if i < col_idx else i - 1) for label, i in col_label_map.items()}
        except KeyError as e:
            print(f"KeyError: {e}. Ensure that both '{label}' and '{merge_with_label}' are in the appropriate label list.")

    merged_row_labels = [label for label in sorted_row_labels if label in row_label_map]
    merged_col_labels = [label for label in sorted_col_labels if label in col_label_map]

    return merged_conf_matrix, merged_row_labels, merged_col_labels


# Example usage:
# sorted_conf_matrix, sorted_row_labels, sorted_col_labels, merge_map = ...  # Your inputs
# merged_conf_matrix, merged_row_labels, merged_col_labels = merge_confusion_matrix(sorted_conf_matrix, sorted_row_labels, sorted_col_labels, merge_map)


def improve_diagonal(conf_matrix, row_labels, col_labels):
    """
    Improve the diagonal elements of the sorted confusion matrix by swapping columns
    if a diagonal element is smaller than another element in the same row, iterating from bottom-right to top-left.

    Parameters:
    conf_matrix (ndarray): The sorted confusion matrix.
    row_labels (list): The sorted labels for rows.
    col_labels (list): The sorted labels for columns.

    Returns:
    improved_conf_matrix (ndarray): The improved confusion matrix with enhanced diagonal elements.
    improved_row_labels (list): The improved labels for rows.
    improved_col_labels (list): The improved labels for columns.
    """
    conf_matrix = conf_matrix.copy()
    col_labels = col_labels.copy()

    for i in range(conf_matrix.shape[0] - 1, -1, -1):
        if i < conf_matrix.shape[1]:
            diag_value = conf_matrix[i, i]
            max_value = diag_value
            max_col = i

            for j in range(conf_matrix.shape[1] - 1, -1, -1):
                if conf_matrix[i, j] > max_value:
                    max_value = conf_matrix[i, j]
                    max_col = j

            if max_col != i:
                # Swap the columns
                conf_matrix[:, [i, max_col]] = conf_matrix[:, [max_col, i]]
                col_labels[i], col_labels[max_col] = col_labels[max_col], col_labels[i]

    improved_conf_matrix = conf_matrix
    improved_row_labels = row_labels  # Row labels remain the same as we are only swapping columns
    improved_col_labels = col_labels

    return improved_conf_matrix, improved_row_labels, improved_col_labels
# Example usage:
# sorted_conf_matrix = np.array([[...], [...], ...])  # Your sorted confusion matrix
# sorted_row_labels = ["row1", "row2", ...]  # Your sorted row labels
# sorted_col_labels = ["col1", "col2", ...]  # Your sorted column labels
# improved_conf_matrix, improved_row_labels, improved_col_labels = improve_diagonal(sorted_conf_matrix, sorted_row_labels, sorted_col_labels)


def confusion_matrix_to_weighted_correlation_matrix(conf_matrix):
    """
    Convert a confusion matrix to a weighted correlation matrix.

    Parameters:
    conf_matrix (ndarray): The confusion matrix to be converted.

    Returns:
    weighted_correlation_matrix (ndarray): The weighted correlation matrix.
    row_heights (ndarray): Heights of the rows.
    col_widths (ndarray): Widths of the columns.
    """
    # Calculate row sums and column sums
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    total_sum = conf_matrix.sum()

    # Calculate row heights and column widths based on their sums
    row_heights = row_sums / total_sum
    col_widths = col_sums / total_sum

    # Normalize the confusion matrix to get percentages
    weighted_correlation_matrix = conf_matrix.astype('float') / total_sum

    return weighted_correlation_matrix, row_heights, col_widths


def calculate_recalls_precisions(correlation_matrix):
    """
    Safely calculate the recalls and precisions of the correlation matrix.

    Parameters:
    correlation_matrix (ndarray): The correlation matrix to calculate recalls and precisions.

    Returns:
    recalls (ndarray): The calculated recalls.
    precisions (ndarray): The calculated precisions.
    """
    min_dim = min(correlation_matrix.shape)
    recalls = np.zeros(min_dim)
    precisions = np.zeros(min_dim)

    for i in range(min_dim):
        row_sum = np.sum(correlation_matrix[i, :])
        col_sum = np.sum(correlation_matrix[:, i])
        diag_elem = correlation_matrix[i, i]

        if row_sum != 0:
            recalls[i] = diag_elem / row_sum * 100
        else:
            recalls[i] = np.nan  # Assign NaN if division by zero

        if col_sum != 0:
            precisions[i] = diag_elem / col_sum * 100
        else:
            precisions[i] = np.nan  # Assign NaN if division by zero

    return recalls, precisions


def visualize_correlation_matrix(correlation_matrix, row_heights, col_widths, row_labels, col_labels,
                                 title='Weighted Correlation Matrix',
                                 ax=None, annot=None, cmap='Blues', offset=0.02, threshold=0.017, merge=None):
    """
    Visualize the weighted correlation matrix.

    Parameters:
    correlation_matrix (ndarray): The weighted correlation matrix to be plotted.
    row_heights (ndarray): Heights of the rows.
    col_widths (ndarray): Widths of the columns.
    labels (list): The labels for the rows and columns.
    title (str): The title of the plot.
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
    annot (bool): Whether to annotate the heatmap with the values.
    cmap (str): Colormap to use for the heatmap.
    """
    if ax is None:
        ax = plt.gca()

    # Normalize the correlation matrix for better contrast
    norm = plt.Normalize(vmin=np.min(correlation_matrix), vmax=np.max(correlation_matrix))

    # Plot the weighted correlation matrix
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            height = row_heights[i]
            width = col_widths[j]
            value = correlation_matrix[i, j]
            if value == 0:
                color = 'white'
            else:
                color = plt.cm.get_cmap(cmap)(norm(value))  # Get color based on value
            rect = plt.Rectangle((sum(col_widths[:j]), sum(row_heights[:i])), width, height,
                                 facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            if annot:
                ax.text(sum(col_widths[:j]) + width / 2, sum(row_heights[:i]) + height / 2,
                        f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', fontsize=10, color='black')

    if merge is None:
        # Add vertical bar for row percentages
        for i in range(correlation_matrix.shape[0]):
            percentage = row_heights[i] * 100
            text_offset = offset if percentage < 0.1 else 0  # Apply offset if percentage is less than 0.1%
            if row_heights[i] > threshold:  # Check if the percentage is greater than zero
                ax.text(1.02, (sum(row_heights[:i]) + row_heights[i] / 2) * ax.get_ylim()[1] - text_offset,
                        f'{percentage:.2f}%', va='center', ha='left', fontsize=10, color='black')

        # Add horizontal bar for column percentages
        for j in range(correlation_matrix.shape[1]):
            percentage = col_widths[j] * 100
            text_offset = offset if percentage < 0.1 else 0  # Apply offset if percentage is less than 0.1%
            if col_widths[j] > threshold:  # Check if the percentage is greater than zero
                ax.text((sum(col_widths[:j]) + col_widths[j] / 2) * ax.get_xlim()[1] + text_offset, -0.05,
                        f'{percentage:.2f}%', va='top', ha='center', fontsize=10, color='black', transform=ax.transAxes,
                        rotation=45)
    else:
        # Calculate recalls and precisions safely
        recalls, precisions = calculate_recalls_precisions(correlation_matrix)
        # Add vertical bars for recalls
        for i, recall in enumerate(recalls):
            if not np.isnan(recall):  # Check if recall is not NaN
                ax.text(1.05, (sum(row_heights[:i]) + row_heights[i] / 2) * ax.get_ylim()[1],
                        f'{recall:.2f}', va='center', ha='left', color='black')

        # Add horizontal bars for precisions
        for j, precision in enumerate(precisions):
            if not np.isnan(precision):  # Check if precision is not NaN
                ax.text((sum(col_widths[:j]) + col_widths[j] / 2) * ax.get_xlim()[1], -0.1,
                        f'{precision:.2f}', va='top', ha='center', color='black', transform=ax.transAxes, rotation=45)

        # Add titles for the recalls and precisions bars
        ax.text(1.05, 0, 'Recalls', va='bottom', ha='left', fontsize=10)
        ax.text(0, -0.05, 'Precisions', va='top', ha='left', fontsize=10, transform=ax.transAxes, rotation=45)
    # Set the ticks and labels
    ax.set_xticks(np.cumsum(col_widths) - col_widths / 2)
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticks(np.cumsum(row_heights) - row_heights / 2)
    ax.set_yticklabels(row_labels)

    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # Invert y-axis to start from the top-left corner
    ax.invert_yaxis()

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')


def visualize_merge_process(cmap, merge_map, ax=None):
    """
    Visualize the merge process using the provided color map and merge map.

    Parameters:
    cmap (ListedColormap): The colormap to use for the visualization.
    merge_map (dict): A map indicating the merges with the format (label, 'row'/'col'): merge_with_label.
    ax (matplotlib.axes.Axes, optional): The axes on which to plot the visualization. Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    merge_instructions = []

    # Visualize column and row merges
    for (label, merge_type), merge_with_label in merge_map.items():
        original_color = cmap.colors[int(label) % len(cmap.colors)]
        merge_color = cmap.colors[int(merge_with_label) % len(cmap.colors)]

        merge_instructions.append((label, merge_with_label, original_color, merge_color, merge_type))

    # Draw background content box
    box = patches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.1", edgecolor="black",
                                 facecolor="lightgrey", lw=2)
    ax.add_patch(box)

    max_rows_per_col = 9
    num_cols = (len(merge_instructions) + max_rows_per_col - 1) // max_rows_per_col  # Calculate the number of columns needed

    for i, (label, merge_with_label, original_color, merge_color, merge_type) in enumerate(merge_instructions):
        col = i // max_rows_per_col  # Determine which column this entry should be in
        row = i % max_rows_per_col  # Determine the row within that column
        y_pos = 0.9 - row * 0.1  # Calculate the y-position based on the row

        x_start = 0.05 + col * 0.5  # Start x position for the column

        if merge_type == 'col':
            ax.text(x_start + 0.02, y_pos, 'Col', fontsize=12, ha='left', va='center')
        else:
            ax.text(x_start + 0.02, y_pos, 'Row', fontsize=12, ha='left', va='center')

        # Draw original color box
        ax.add_patch(patches.Rectangle((x_start + 0.1, y_pos - 0.05), 0.1, 0.1, color=original_color))
        ax.text(x_start + 0.15, y_pos, label, fontsize=12, ha='center', va='center', color='white', fontweight='bold')

        # Draw merge color box
        ax.add_patch(patches.Rectangle((x_start + 0.3, y_pos - 0.05), 0.1, 0.1, color=merge_color))
        ax.text(x_start + 0.35, y_pos, merge_with_label, fontsize=12, ha='center', va='center', color='white', fontweight='bold')

        # Draw arrow
        ax.annotate('', xy=(x_start + 0.25, y_pos), xytext=(x_start + 0.2, y_pos),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def calculate_metrics(ground_truth, labels):
    try:
        label = np.sort(np.union1d(np.unique(ground_truth), np.unique(labels)))
        conf_matrix = confusion_matrix(ground_truth, labels, labels=label)
        if label[0] == 0:
            adjusted_conf_matrix = np.hstack((conf_matrix[:, 1:], conf_matrix[:, 0:1]))
            adjusted_label = np.hstack((label[1:], label[0]))
            reduced_conf_matrix = remove_trailing_zero_rows(adjusted_conf_matrix)
        else:
            adjusted_conf_matrix = conf_matrix
            adjusted_label = label
            reduced_conf_matrix = remove_trailing_zero_rows(adjusted_conf_matrix)
        # print(conf_matrix)
        # print(reduced_conf_matrix)
        # print(adjusted_conf_matrix)
        OA = np.trace(reduced_conf_matrix) / np.sum(reduced_conf_matrix)
        per_class_acc = reduced_conf_matrix.diagonal() / reduced_conf_matrix.sum(axis=1)
        AA = np.mean(per_class_acc)
        Kappa = cohen_kappa_score(ground_truth, labels)
        ARI = adjusted_rand_score(ground_truth, labels)
        return OA, AA, Kappa, ARI, adjusted_conf_matrix, adjusted_label
    except Exception as e:
        print(e)


def visualize_results(name, hsi_data, ground_truth, labels_best, param_metrics, dataset_name, adjusted_conf_matrix,
                      adjusted_labels, method_name, feature_para_pair, best_params, OA, AA, Kappa, ARI,
                      threshold=0.017):
    h, w = hsi_data.shape[:2]
    clustered_image = labels_best.reshape((h, w))
    ground_truth_image = ground_truth.reshape((h, w))
    masked_image = np.where(ground_truth_image > 0, clustered_image, np.nan)
    masked_gt = np.where(ground_truth_image > 0, ground_truth_image, np.nan)

    fig, axs = plt.subplots(3, 3, figsize=(18, 18))

    axs[0, 0].imshow(masked_gt, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[0, 0].set_title(f'Ground Truth ({dataset_name})')

    axs[0, 1].imshow(clustered_image, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[0, 1].set_title(f'Best PCA + {method_name} ({dataset_name})')

    axs[0, 2].imshow(masked_image, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[0, 2].set_title(f'Masked PCA + {method_name} ({dataset_name})')

    # # Assuming param_metrics and feature_para_pair are defined and contain the necessary data
    # df = pd.DataFrame(param_metrics)
    # # df = df[df['pca_components'].notna()]
    # x = str(feature_para_pair[0])
    # y = str(feature_para_pair[1])
    # df = df[(df[x].notna()) & (df[y].notna())]
    # xi = np.linspace(df[x].min(), df[x].max(), 100)
    # yi = np.linspace(df[y].min(), df[y].max(), 100)
    # xi, yi = np.meshgrid(xi, yi)
    # zi = griddata((df[x], df[y]), df['ARI'], (xi, yi), method='cubic')
    #
    # heatmap = axs[1, 0].imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='viridis',
    #                            aspect='auto')
    # axs[1, 0].set_title('Parameter Tuning Heatmap (ARI)')
    # axs[1, 0].set_xlabel(x)
    # axs[1, 0].set_ylabel(y)
    # fig.colorbar(heatmap, ax=axs[1, 0], shrink=0.5, aspect=5)
    # Assuming param_metrics and feature_para_pair are defined and contain the necessary data
    df = pd.DataFrame(param_metrics)
    # df = df[df['pca_components'].notna()]
    x = str(feature_para_pair[0])
    y = str(feature_para_pair[1])
    df = df[(df[x].notna()) & (df[y].notna())]
    # Create a pivot table
    pivot_table = df.pivot_table(index=y, columns=x, values='ARI', aggfunc='mean')
    # Plot the block map
    heatmap = axs[1, 0].imshow(pivot_table, origin='lower', cmap='viridis', aspect='auto')
    # Set the x and y ticks to correspond to the actual parameter values
    axs[1, 0].set_xticks(np.arange(len(pivot_table.columns)))
    axs[1, 0].set_yticks(np.arange(len(pivot_table.index)))
    axs[1, 0].set_xticklabels(pivot_table.columns)
    axs[1, 0].set_yticklabels(pivot_table.index)
    axs[1, 0].set_title('Parameter Tuning Heatmap (ARI)')
    axs[1, 0].set_xlabel(x)
    axs[1, 0].set_ylabel(y)
    fig.colorbar(heatmap, ax=axs[1, 0], shrink=0.5, aspect=5)

    sns.heatmap(adjusted_conf_matrix, annot=False, fmt='d', cmap='Blues',
                xticklabels=adjusted_labels,
                yticklabels=adjusted_labels, ax=axs[1, 1])
    # Add a red square
    a = len(adjusted_labels)
    b = remove_trailing_zero_rows(adjusted_conf_matrix).shape[0]
    rect = Rectangle((adjusted_labels[0] - 1, adjusted_labels[0] - 1), a, b, fill=False, edgecolor='blue', linewidth=2)
    axs[1, 1].plot([0, a], [0, a], color='red', linestyle='--', linewidth=2)
    axs[1, 1].add_patch(rect)
    axs[1, 1].set_title('Confusion Matrix')
    axs[1, 1].set_xlabel('Clustered Sets')
    axs[1, 1].set_ylabel('Ground-Truth Sets')
    axs[1, 1].xaxis.set_label_position('top')
    axs[1, 1].xaxis.tick_top()

    # Define the text to display
    text = (
        f"{method_name} Best Parameters for {dataset_name}:\n{', '.join(map(str, best_params.values()))}\n"
        f'Corr. OA: {OA:.4f}\n'
        f'Corr. AA: {AA:.4f}\n'
        f'Corr. Ka: {Kappa:.4f}\n'
        f'Best ARI: {ARI:.4f}'
    )
    axs[1, 2].text(0.5, 0.5, text,
                   fontsize=14,  # Larger font size
                   fontfamily='monospace',  # Monospace font for uniform character width
                   color='darkblue',  # Dark blue color
                   weight='bold',  # Bold font
                   va='center', ha='center',  # Centered alignment
                   bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=1')  # Background box
                   )
    axs[1, 2].axis('off')

    reordered_conf_matrix, merge_map, reordered_row_labels, reordered_col_labels = optimize_diagonal_elements(
        adjusted_conf_matrix, adjusted_labels, adjusted_labels, thrshold=threshold)
    print(merge_map)
    # Sort the confusion matrix
    sorted_conf_matrix, sorted_row_labels, sorted_col_labels = sort_confusion_matrix(reordered_conf_matrix,
                                                                                     reordered_row_labels,
                                                                                     reordered_col_labels, merge_map)
    # Convert the sorted confusion matrix to a weighted correlation matrix
    weighted_correlation_matrix, row_heights, col_widths = confusion_matrix_to_weighted_correlation_matrix(
        sorted_conf_matrix)
    # Visualize the weighted correlation matrix on the specified subplot
    visualize_correlation_matrix(weighted_correlation_matrix, row_heights, col_widths, sorted_row_labels,
                                 sorted_col_labels, title='Weighted Correlation Matrix', ax=axs[2, 0],
                                 threshold=threshold)
    df = pd.DataFrame(sorted_conf_matrix)
    df.to_csv(f'{method_name}_{name}_conf_matrix.csv', index=False)

    merged_conf_matrix, merged_row_labels, merged_col_labels = merge_confusion_matrix(sorted_conf_matrix,
                                                                                      sorted_row_labels,
                                                                                      sorted_col_labels, merge_map)
    improved_conf_matrix, improved_row_labels, improved_col_labels = improve_diagonal(merged_conf_matrix,
                                                                                      merged_row_labels,
                                                                                      merged_col_labels)
    # Convert the sorted confusion matrix to a weighted correlation matrix
    weighted_merged_matrix, row_heights, col_widths = confusion_matrix_to_weighted_correlation_matrix(
        improved_conf_matrix)
    # Visualize the weighted correlation matrix
    visualize_correlation_matrix(improved_conf_matrix, row_heights, col_widths, improved_row_labels,
                                 improved_col_labels, title='Weighted Correlation Matrix', merge=True, ax=axs[2, 2],
                                 threshold=threshold)
    df = pd.DataFrame(improved_conf_matrix)
    df.to_csv(f'{method_name}_{name}_conf_matrix_merged.csv', index=False)

    visualize_merge_process(cmap, merge_map, ax=axs[2, 1])

    plt.tight_layout()
    plt.savefig(f'{method_name}_{dataset_name}.png', format='png')
    plt.show()


def reshape_with_spatial_embedding(hsi_data):
    """
    Reshape the HSI data with spatial embedding.

    Parameters:
    hsi_data (ndarray): The input HSI data with shape (height, width, spectral).

    Returns:
    reshaped_data (ndarray): The reshaped HSI data with spatial embedding.
    """
    num_samples = hsi_data.shape[0] * hsi_data.shape[1]
    num_features = hsi_data.shape[2]
    reshaped_data_i = []

    for x in range(hsi_data.shape[0]):
        for y in range(hsi_data.shape[1]):
            # Get the original data for the pixel
            pixel_data = list(hsi_data[x, y, :])
            # Append the spatial coordinates
            pixel_data.append(x)
            pixel_data.append(y)
            # Append this modified data to the reshaped data
            reshaped_data_i.append(pixel_data)

    reshaped_data_i = np.array(reshaped_data_i)
    return reshaped_data_i


def process_dataset(name, hsi_data, ground_truth, method_name, param_grid, feature_para_pair, max_workers=1,
                    no_tuning=None, threshold=0.017):
    if method_name == "kNN":
        unique_combinations = {(params['ksize'], params['spatial_embedding']) for params in ParameterGrid(param_grid)}
        preprocessed_data_dict = {}
        for kernel_size, spatial_embedding in unique_combinations:
            preprocessed_data = preprocess_hsi_data(hsi_data, kernel_size)
            if spatial_embedding:
                preprocessed_data = reshape_with_spatial_embedding(preprocessed_data)
            else:
                num_samples = hsi_data.shape[0] * hsi_data.shape[1]
                num_features = hsi_data.shape[2]
                preprocessed_data = preprocessed_data.reshape((num_samples, num_features))
            preprocessed_data_dict[(kernel_size, spatial_embedding)] = preprocessed_data

        def evaluate_knn_params(params):
            kernel_size = params.get('ksize', None)
            spatial_embedding = params.get('spatial_embedding', False)
            preprocessed_data = preprocessed_data_dict[(kernel_size, spatial_embedding)]
            pca_data = apply_pca(preprocessed_data, n_components=params['pca_components'])

            # Setting the radius to be infinite
            radius = float('inf')

            knn = NearestNeighbors(n_neighbors=params['n_neighbors'] + 1, n_jobs=-1, radius=radius)
            knn.fit(pca_data)
            distances, indices = knn.kneighbors(pca_data)

            # Ensure the first neighbor is the point itself and exclude it
            if not np.all(distances[:, 0] == 0):
                raise ValueError("Self-distances are not zero. There might be an issue with data preprocessing or PCA.")

            distances = distances[:, :params['n_neighbors'] + 1]
            indices = indices[:, :params['n_neighbors'] + 1]

            # Create strings for the filename
            spatial_str = "spatial_embedding" if spatial_embedding else "non_embedding"
            pca_str = f'pca_{params["pca_components"]}'
            kernel_str = f'kernel_{kernel_size}' if kernel_size else 'no_kernel'

            # Save distances and indices as k x n matrices using pandas
            distances_df = pd.DataFrame(distances)
            indices_df = pd.DataFrame(indices)
            distances_df.to_csv(f'{name}_knn_{spatial_str}_{pca_str}_{kernel_str}_distances.csv', index=False, header=False)
            indices_df.to_csv(f'{name}_knn_{spatial_str}_{pca_str}_{kernel_str}_indices.csv', index=False, header=False)
            return None, None, None, None, None, None, None, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_knn_params, params) for params in ParameterGrid(param_grid)]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing kNN for {name}"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error with parameters: {e}")
        return

    best_score = -np.inf
    best_params = None
    best_labels = None
    param_metrics = []
    best_adjusted_conf_matrix = None
    best_adjusted_labels = None

    # Get unique combinations of kernel sizes and spatial embedding from the parameter grid
    unique_combinations = {(params['ksize'], params['spatial_embedding']) for params in ParameterGrid(param_grid)}

    # Preprocess HSI data for each unique combination
    preprocessed_data_dict = {}
    for kernel_size, spatial_embedding in unique_combinations:
        preprocessed_data = preprocess_hsi_data(hsi_data, kernel_size)
        if spatial_embedding:
            preprocessed_data = reshape_with_spatial_embedding(preprocessed_data)
        else:
            num_samples = hsi_data.shape[0] * hsi_data.shape[1]
            num_features = hsi_data.shape[2]
            preprocessed_data = preprocessed_data.reshape((num_samples, num_features))
        preprocessed_data_dict[(kernel_size, spatial_embedding)] = preprocessed_data

    def evaluate_params(params):

        # Get preprocessed HSI data for the specified kernel size and spatial embedding
        kernel_size = params.get('ksize', None)
        spatial_embedding = params.get('spatial_embedding', False)
        preprocessed_data = preprocessed_data_dict[(kernel_size, spatial_embedding)]

        # PCA
        pca_data = apply_pca(preprocessed_data, n_components=params['pca_components'])

        # Clustering
        switch = {
            "GMM": lambda: cluster_gmm(pca_data, n_components=params['gmm_components'],
                                       covariance_type=params['covariance_type'], init_params=params['init_params'],
                                       max_iter=params['max_iter'], random_state=params['random_state']),
            "DBSCAN": lambda: cluster_dbscan(pca_data, epsilon=params['epsilon'], minPts=params['minPts']),
            "HDBSCAN": lambda: cluster_hdbscan(pca_data, min_samples=params['min_samples'],
                                               min_cluster_size=params['min_cluster_size']),
            "Spectral": lambda: cluster_spectral(pca_data, n_clusters=params['n_clusters'], affinity=params['affinity'],
                                                 gamma=params.get('gamma', None),
                                                 n_neighbors=params.get('n_neighbors', None),
                                                 n_components=params['n_components'],
                                                 assign_labels=params['assign_labels']),
            "KMeans": lambda: cluster_kmeans(pca_data, n_clusters=params['n_clusters'], init=params['init'],
                                             n_init=params['n_init'], max_iter=params['max_iter'], tol=params['tol'],
                                             random_state=params['random_state']),

        }
        labels = switch[method_name]()

        # Matching
        label_map = match_labels(ground_truth, labels)
        matched_labels = np.array([label_map[label] for label in labels])

        # Mask
        labeled_mask = ground_truth > 0
        labeled_ground_truth = ground_truth[labeled_mask]
        masked_labels = matched_labels[labeled_mask]

        # Metrics
        OA, AA, Kappa, ARI, adjusted_conf_matrix, adjusted_labels = calculate_metrics(labeled_ground_truth,
                                                                                      masked_labels)
        metric = {**params, **{'OA': OA, 'AA': AA, 'Kappa': Kappa, 'ARI': ARI}}
        param_metrics.append(metric)

        return OA, AA, Kappa, ARI, params, adjusted_conf_matrix, adjusted_labels, matched_labels

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params, params) for params in ParameterGrid(param_grid)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tuning {name}"):
            try:
                OA, AA, Kappa, ARI, params, adjusted_conf_matrix, adjusted_labels, matched_labels = future.result()
                if ARI > best_score:
                    best_score = ARI
                    best_labels = matched_labels
                    best_OA = OA
                    best_AA = AA
                    best_Kappa = Kappa
                    best_params = params
                    best_adjusted_conf_matrix = adjusted_conf_matrix
                    best_adjusted_labels = adjusted_labels
            except Exception as e:
                if params:
                    print(f"Error with parameters {params}: {e}")
                else:
                    print(f"Error: {e}")
    if not no_tuning:
        # Save the parameter metrics to a CSV file
        df = pd.DataFrame(param_metrics)
        df.to_csv(f'{method_name}_{name}_tuning.csv', index=False)

        # df = pd.DataFrame(best_adjusted_conf_matrix)
        # df.to_csv(f'{method_name}_{name}_conf_matrix.csv', index=False)

        # df = pd.DataFrame(best_adjusted_labels)
        # df.to_csv(f'{method_name}_{name}_labels.csv', index=False)
        df = pd.DataFrame(best_labels)
        df.to_csv(f'{method_name}_{name}.csv', index=False)
    else:
        param_metrics = pd.read_csv(f'{method_name}_{name}_tuning.csv', index_col=0)

    visualize_results(name, hsi_data, ground_truth, best_labels, param_metrics, name, best_adjusted_conf_matrix,
                      best_adjusted_labels, method_name, feature_para_pair, best_params, best_OA, best_AA, best_Kappa,
                      best_score, threshold=threshold)