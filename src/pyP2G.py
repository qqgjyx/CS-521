import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import igraph as ig
import leidenalg as la
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.pyC4H import (colors, remove_trailing_zero_rows, optimize_diagonal_elements, sort_confusion_matrix,
                   confusion_matrix_to_weighted_correlation_matrix, visualize_correlation_matrix, visualize_merge_process,
                   merge_confusion_matrix, improve_diagonal, match_labels, calculate_metrics, load_hsi_data, cmap)

from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import lil_matrix, csr_matrix

def plot_graph(A, title, layout='spring', directed=True, node_size=20, alpha=0.3, arrowsize=10, seed=42, membership=None):
    # Choose the appropriate graph type based on the 'directed' flag
    if directed:
        G = nx.DiGraph()  # Directed graph
    else:
        G = nx.Graph()  # Undirected graph

    n = len(A)

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    # Choose the layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed)  # Use a fixed seed for reproducibility
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout type: {layout}")

    plt.figure(figsize=(12, 12))  # Increase figure size to avoid label overlap

    # Assign colors based on membership
    if membership is not None:
        unique_membership = set(membership)
        color_map = plt.cm.get_cmap('viridis', len(unique_membership))
        node_colors = [color_map(membership[i]) for i in range(n)]
    else:
        node_colors = 'blue'  # Default color if no membership is provided

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)  # Smaller node size

    if directed:
        nx.draw_networkx_edges(G, pos, alpha=alpha, arrowstyle='-|>', arrowsize=arrowsize)  # Directed edges with arrows
    else:
        nx.draw_networkx_edges(G, pos, alpha=alpha)  # Undirected edges without arrows

    plt.title(title)
    plt.show()

# Example usage:
# plot_graph(A, "Graph Visualization", layout='spring', directed=True)  # For a directed graph
# plot_graph(A, "Graph Visualization", layout='spring', directed=False)  # For an undirected graph


def plot_tsne(A, title, n_components=3, perplexity=30, learning_rate=200, n_iter=1000, normalize_data=False,
              membership=None):
    # Optional: Normalize the adjacency matrix
    if normalize_data:
        A = normalize(A, norm='l2')

    # Apply t-SNE to the (possibly normalized) adjacency matrix
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=n_iter,
                random_state=42)
    tsne_results = tsne.fit_transform(A)

    # Create a plot based on the number of components
    if n_components == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Assign colors based on membership
        if membership is not None:
            unique_membership = set(membership)
            color_map = plt.cm.get_cmap('viridis', len(unique_membership))
            node_colors = [color_map(membership[i]) for i in range(len(tsne_results))]
        else:
            node_colors = 'blue'  # Default color if no membership is provided

        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], s=20, c=node_colors)

        # Optionally, you can also plot edges if desired
        G = nx.from_numpy_array(A)
        for i, j in G.edges():
            ax.plot([tsne_results[i, 0], tsne_results[j, 0]],
                    [tsne_results[i, 1], tsne_results[j, 1]],
                    [tsne_results[i, 2], tsne_results[j, 2]], 'k-', alpha=0.01)
    elif n_components == 2:
        plt.figure(figsize=(10, 10))

        # Assign colors based on membership
        if membership is not None:
            unique_membership = set(membership)
            color_map = plt.cm.get_cmap('viridis', len(unique_membership))
            node_colors = [color_map(membership[i]) for i in range(len(tsne_results))]
        else:
            node_colors = 'blue'  # Default color if no membership is provided

        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=20, c=node_colors)

        # Optionally, you can also plot edges if desired
        G = nx.from_numpy_array(A)
        for i, j in G.edges():
            plt.plot([tsne_results[i, 0], tsne_results[j, 0]],
                     [tsne_results[i, 1], tsne_results[j, 1]], 'k-', alpha=0.01)

    else:
        raise ValueError(f"n_components={n_components} is not supported. Use 2 or 3.")

    plt.title(title)
    plt.show()


# Example usage with 2D and 3D options:
# plot_tsne(A, "3D t-SNE of Graph", n_components=3, perplexity=50, learning_rate=100, n_iter=3000, normalize_data=True)
# plot_tsne(A, "2D t-SNE of Graph", n_components=2, perplexity=30, learning_rate=200, n_iter=1000, normalize_data=False)


def process_node(i, indices, distances, k, distribution, sigma, df):
    local_sigma = sigma
    if distribution == 'adaptive_gaussian':
        local_sigma = np.mean(distances[i][1:k + 1])

    weights = []
    for j in range(1, k + 1):  # start from 1 to exclude the point itself
        neighbor_index = indices[i][j]
        distance = distances[i][j]

        # Determine weight based on the specified distribution
        if distribution == 'gaussian':
            weight = np.exp(-distance ** 2 / (2 * sigma ** 2))
        elif distribution == 'adaptive_gaussian':
            weight = np.exp(-distance ** 2 / (2 * local_sigma ** 2))
        elif distribution == 't_distribution':
            weight = (1 + distance ** 2 / df) ** -(df + 1) / 2
        else:
            raise ValueError("Unsupported distribution type")

        weights.append((neighbor_index, i, weight))

    return weights

def create_knn_graph(indices, distances, k, symmetrize=False, distribution='adaptive_gaussian', sigma=1.0, df=1.0):
    n = len(indices)
    A = lil_matrix((n, n))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_node, i, indices, distances, k, distribution, sigma, df) for i in range(n)]
        for future in futures:
            weights = future.result()
            for neighbor_index, i, weight in weights:
                A[neighbor_index, i] = weight  # Ensure directed edge: neighbor_index -> i

    if symmetrize:
        # Symmetrize the matrix by averaging weights for mutual neighbors
        A = (A + A.T) / 2

    # Normalize the matrix to be column-stochastic
    A = A.tocsr()
    column_sums = np.array(A.sum(axis=0)).flatten()
    column_sums[column_sums == 0] = 1  # Prevent division by zero
    A = A.multiply(1 / column_sums)

    # Calculate sparsity
    sparsity = 1 - (A.nnz / (n * n))
    print(f"Sparsity of the adjacency matrix: {sparsity}")

    return A


def assert_graph_requirements(A, k, tolerance=1e-15):
    passed = True
    results = []

    # Check that all weights are positive where the adjacency matrix has non-zero entries
    if (A.data >= -tolerance).all():
        results.append("1. All weights are positive: PASS")
    else:
        results.append("1. All weights are positive: FAIL")
        print("Negative weights found at the following positions:")
        negative_positions = np.argwhere(A.data < -tolerance)
        for pos in negative_positions:
            print(f"Position: {pos}, Value: {A.data[pos]}")
        passed = False

    # Check that each column has exactly k non-zero elements
    non_zero_elements_per_column = (A > tolerance).sum(axis=0).A1
    if np.allclose(non_zero_elements_per_column, k):
        results.append(f"2. Each column has exactly {k} non-zero elements: PASS")
    else:
        results.append(f"2. Each column has exactly {k} non-zero elements: FAIL")
        print("Columns with incorrect number of non-zero elements:")
        incorrect_columns = np.where(~np.isclose(non_zero_elements_per_column, k))[0]
        for col in incorrect_columns:
            print(f"Column: {col}, Non-zero elements: {non_zero_elements_per_column[col]}")
        passed = False

    # Check that the matrix is column-stochastic
    column_sums = A.sum(axis=0).A1
    if np.allclose(column_sums, 1, atol=tolerance):
        results.append("3. Matrix is column-stochastic: PASS")
    else:
        results.append("3. Matrix is column-stochastic: FAIL")
        print("Columns that are not column-stochastic (sum != 1):")
        for col in np.where(~np.isclose(column_sums, 1, atol=tolerance))[0]:
            print(f"Column: {col}, Sum: {column_sums[col]}")
        passed = False

    # Print results summary
    for result in results:
        print(result)

    if not passed:
        raise AssertionError("One or more graph requirements were not met.")


def leiden_community_detection(A, resolution_parameter=1, output_dir='./', dataset_name=None):
    """
    Perform Leiden community detection on a graph represented by an adjacency matrix.

    Parameters:
    A (scipy.sparse.coo_matrix): The adjacency matrix of the graph.
    resolution_parameter (float): The resolution parameter for the Leiden algorithm.
    output_dir (str): Directory to save the output CSV file.

    Returns:
    membership_vector (list): The membership vector indicating community assignments.
    modularity_score (float): The modularity score of the detected partition.
    """
    # Convert the sparse matrix to an igraph graph
    sources, targets = A.nonzero()
    weights = A.data
    edges = list(zip(sources, targets))
    g = ig.Graph(edges=edges, directed=True)
    g.es['weight'] = weights.tolist()

    # Run the Leiden algorithm for community detection
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter)

    # Extract the membership vector
    membership_vector = partition.membership

    # Get the modularity score
    modularity_score = partition.modularity

    # Save the membership vector to a CSV file
    output_filename = f"{dataset_name}_membership_resolution_{resolution_parameter}.csv"
    output_path = output_dir + output_filename
    pd.DataFrame(membership_vector, columns=["Community"]).to_csv(output_path, index=False)
    print(f"Saved membership vector to {output_path}")

    # Append the modularity score to a CSV file
    modularity_csv = output_dir + f"{dataset_name}_modularity_scores.csv"
    df = pd.DataFrame({"Resolution": [resolution_parameter], "Modularity": [modularity_score]})

    # Append to the CSV file if it exists, otherwise create it
    df.to_csv(modularity_csv, mode='a', header=not pd.io.common.file_exists(modularity_csv), index=False)
    print(f"Appended modularity score to {modularity_csv}")

    return membership_vector, modularity_score


def process_membership_vectors(dataset_name, hsi_data, ground_truth, membership_vector, adjusted_conf_matrix,
                               adjusted_labels, resolution1,
                               membership_vector2=None, resolution2=None, method_name="Leiden",
                               threshold=0.017, cmap=None):
    name = dataset_name
    h, w = hsi_data.shape[:2]
    # Step 1: Reshape Membership Vectors
    clustered_image = membership_vector.reshape((h, w))
    ground_truth_image = ground_truth.reshape((h, w))
    masked_gt = np.where(ground_truth_image > 0, ground_truth_image, np.nan)
    masked_image = np.where(ground_truth_image > 0, clustered_image, np.nan)

    if membership_vector2 is not None:
        clustered_image2 = membership_vector2.reshape((h, w))
        masked_image2 = np.where(ground_truth_image > 0, clustered_image2, np.nan)
    else:
        clustered_image2 = None
        masked_image2 = None

    # Step 2: Visualization Setup
    # If second membership vector is provided, create additional subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))

    axs[0, 0].imshow(masked_gt, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[0, 0].set_title(f'Ground Truth ({name})')

    axs[1, 1].imshow(clustered_image, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[1, 1].set_title(f'Partition (Res={resolution1})')

    axs[0, 1].imshow(masked_image, cmap=cmap, vmin=0, vmax=len(colors) - 1)
    axs[0, 1].set_title(f'Masked Partition (Res={resolution1})')

    if membership_vector2 is not None:
        axs[1, 2].imshow(clustered_image2, cmap=cmap, vmin=0, vmax=len(colors) - 1)
        axs[1, 2].set_title(f'Partition 2 (Res={resolution2})')

        axs[0, 2].imshow(masked_image2, cmap=cmap, vmin=0, vmax=len(colors) - 1)
        axs[0, 2].set_title(f'Masked Partition 2 (Res={resolution2})')

    # Step 5: Confusion Matrix Visualization
    if (len(adjusted_labels) > 20):
        annot = False
    else:
        annot = True
    sns.heatmap(adjusted_conf_matrix, annot=annot, fmt='d', cmap='Blues', xticklabels=adjusted_labels,
                yticklabels=adjusted_labels, ax=axs[1, 0])
    axs[1, 0].set_title('Adjusted Confusion Matrix')
    axs[1, 0].set_xlabel(
        f'Partition (Res={resolution1})' if membership_vector2 is None else f'Partition 2 (Res={resolution2})')
    axs[1, 0].set_ylabel('Ground Truth Clusters' if membership_vector2 is None else f'Partition 1 (Res={resolution1})')
    # Add a red square
    a = len(adjusted_labels)
    b = remove_trailing_zero_rows(adjusted_conf_matrix).shape[0]
    rect = Rectangle((adjusted_labels[0] - 1, adjusted_labels[0] - 1), a, b, fill=False, edgecolor='blue', linewidth=2)
    axs[1, 0].plot([0, a], [0, a], color='red', linestyle='--', linewidth=2)
    axs[1, 0].add_patch(rect)
    axs[1, 0].xaxis.set_label_position('top')
    axs[1, 0].xaxis.tick_top()

    # Step 6: Correlation Matrix Visualization
    reordered_conf_matrix, merge_map, reordered_row_labels, reordered_col_labels = optimize_diagonal_elements(
        adjusted_conf_matrix, adjusted_labels, adjusted_labels, thrshold=threshold)
    sorted_conf_matrix, sorted_row_labels, sorted_col_labels = sort_confusion_matrix(reordered_conf_matrix,
                                                                                     reordered_row_labels,
                                                                                     reordered_col_labels, merge_map)
    weighted_correlation_matrix, row_heights, col_widths = confusion_matrix_to_weighted_correlation_matrix(
        sorted_conf_matrix)
    visualize_correlation_matrix(sorted_conf_matrix, row_heights, col_widths, sorted_row_labels, sorted_col_labels,
                                 title='Weighted Correlation Matrix',
                                 ax=axs[2, 0], threshold=threshold)
    axs[2, 0].set_xlabel(
        f'Partition (Res={resolution1})' if membership_vector2 is None else f'Partition 2 (Res={resolution2})')
    axs[2, 0].set_ylabel('Ground Truth Clusters' if membership_vector2 is None else f'Partition 1 (Res={resolution1})')

    # Step 7: Merge Process Visualization
    visualize_merge_process(cmap, merge_map, ax=axs[2, 1])

    # Step 8: Improved Weighted Confusion Matrix
    merged_conf_matrix, merged_row_labels, merged_col_labels = merge_confusion_matrix(sorted_conf_matrix,
                                                                                      sorted_row_labels,
                                                                                      sorted_col_labels, merge_map)
    improved_conf_matrix, improved_row_labels, improved_col_labels = improve_diagonal(merged_conf_matrix,
                                                                                      merged_row_labels,
                                                                                      merged_col_labels)
    weighted_merged_matrix, row_heights, col_widths = confusion_matrix_to_weighted_correlation_matrix(
        improved_conf_matrix)
    visualize_correlation_matrix(improved_conf_matrix, row_heights, col_widths, improved_row_labels,
                                 improved_col_labels,
                                 title='Improved Weighted Correlation Matrix', merge=True,
                                 ax=axs[2, 2], threshold=threshold)
    axs[2, 2].set_xlabel(
        f'Partition (Res={resolution1})' if membership_vector2 is None else f'Partition 2 (Res={resolution2})')
    axs[2, 2].set_ylabel('Ground Truth Clusters' if membership_vector2 is None else f'Partition 1 (Res={resolution1})')

    plt.tight_layout()
    plt.savefig(f'{method_name}_{name}_results_res{resolution1}_{"res" + str(resolution2) if resolution2 else ""}.png',
                format='png')
    plt.show()


def load_membership_vector_by_resolution(resolution_parameter, dataset_name):
    """
    Load a membership vector from a CSV file based on the resolution parameter.

    Parameters:
    resolution_parameter (float): The resolution parameter used to generate the filename.

    Returns:
    pd.Series: The loaded membership vector.
    """
    # Generate the filename based on the resolution parameter
    file_name = f"{dataset_name}_membership_resolution_{resolution_parameter}.csv"

    # Load the membership vector from the generated filename
    membership_vector = pd.read_csv(file_name).squeeze()
    membership_vector = np.array(membership_vector, dtype='uint8')
    return membership_vector


def process_resolutions(dataset_name, hsi_data, ground_truth, resolution1, resolution2=None, method_name="Leiden", threshold=0.017, cmap=None):
    """
    Process membership vectors for the given resolutions, including one resolution or two resolutions for comparison.

    Parameters:
    hsi_data (ndarray): The hyperspectral image data.
    ground_truth (ndarray): The ground truth labels.
    resolution1 (float): The first resolution parameter.
    resolution2 (float, optional): The second resolution parameter for comparison. Defaults to None.
    method_name (str): The name of the clustering method used. Defaults to "Leiden".
    cmap (ListedColormap): The colormap to use for visualizations. Defaults to None.
    """
    # Load the membership vectors based on the resolutions
    membership_vector = load_membership_vector_by_resolution(resolution1, dataset_name)

    # Matching for resolution1
    label_map = match_labels(ground_truth, membership_vector)
    matched_labels = np.array([label_map[label] for label in membership_vector])

    # Mask for resolution1
    labeled_mask = ground_truth > 0
    labeled_ground_truth = ground_truth[labeled_mask]
    masked_labels = matched_labels[labeled_mask]

    # Metrics for resolution1
    OA, AA, Kappa, ARI, adjusted_conf_matrix, adjusted_labels = calculate_metrics(labeled_ground_truth, masked_labels)

    # Visualization for resolution1
    process_membership_vectors(
        dataset_name=dataset_name,
        hsi_data=hsi_data,
        ground_truth=ground_truth,
        membership_vector=matched_labels,
        adjusted_conf_matrix=adjusted_conf_matrix,
        adjusted_labels=adjusted_labels,
        resolution1=resolution1,
        membership_vector2=None,
        resolution2=None,
        method_name=method_name,
        threshold=threshold,
        cmap=cmap
    )

    # If resolution2 is provided, process it as well
    if resolution2 is not None:
        membership_vector2 = load_membership_vector_by_resolution(resolution2, dataset_name)

        # Matching for resolution2
        label_map2 = match_labels(ground_truth, membership_vector2)
        matched_labels2 = np.array([label_map2[label] for label in membership_vector2])

        # Mask for resolution2
        masked_labels2 = matched_labels2[labeled_mask]

        # Metrics for resolution2
        OA2, AA2, Kappa2, ARI2, adjusted_conf_matrix2, adjusted_labels2 = calculate_metrics(labeled_ground_truth,
                                                                                            masked_labels2)

        # Visualization for resolution2
        process_membership_vectors(
            dataset_name=dataset_name,
            hsi_data=hsi_data,
            ground_truth=ground_truth,
            membership_vector=matched_labels2,
            adjusted_conf_matrix=adjusted_conf_matrix2,
            adjusted_labels=adjusted_labels2,
            resolution1=resolution2,
            membership_vector2=None,
            resolution2=None,
            method_name=method_name,
            threshold=threshold,
            cmap=cmap
        )

        # Matching between resolution1 and resolution2
        label_map1_2 = match_labels(matched_labels, matched_labels2)
        matched_labels1_2 = np.array([label_map1_2[label] for label in matched_labels2])

        # Mask for combined resolution1 and resolution2
        masked_labels1 = matched_labels[labeled_mask]
        masked_labels1_2 = matched_labels1_2[labeled_mask]

        # Metrics for combined resolution1 and resolution2
        OA1_2, AA1_2, Kappa1_2, ARI1_2, adjusted_conf_matrix1_2, adjusted_labels1_2 = calculate_metrics(masked_labels1,
                                                                                                        masked_labels1_2)

        # Visualization for combined resolution1 and resolution2
        process_membership_vectors(
            dataset_name=dataset_name,
            hsi_data=hsi_data,
            ground_truth=ground_truth,
            membership_vector=matched_labels,
            adjusted_conf_matrix=adjusted_conf_matrix1_2,
            adjusted_labels=adjusted_labels1_2,
            resolution1=resolution1,
            membership_vector2=matched_labels1_2,
            resolution2=resolution2,
            method_name=method_name,
            threshold=threshold,
            cmap=cmap
        )

# Example usage
# Assuming hsi_data, ground_truth have been loaded, and cmap is defined
# process_resolutions(hsi_data, ground_truth, resolution1=1.0, resolution2=2.0, method_name="Leiden", cmap=cmap)