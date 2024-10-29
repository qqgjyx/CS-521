import numpy as np

def get_best_params(df, dataset_name, algorithm_name):
    # Filter for the dataset and algorithm
    filtered_df = df[(df['dataset'] == dataset_name) & (df['algorithm'] == algorithm_name)]
    if filtered_df.empty:
        return None

    # Get row with best ARI
    best_row = filtered_df.loc[filtered_df['ARI'].idxmax()]

    # Convert string params to dict
    import ast
    params = ast.literal_eval(best_row['params'])

    return {
        'params': params,
        'ari': best_row['ARI'],
        'n_clusters': best_row['n_clusters']
    }


def get_datasets():
    from sklearn import datasets
    n_samples = 500
    seed = 30
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    rng = np.random.RandomState(seed)
    # ------------------------------------------------------------------------------------------------------------------
    # add the labels to be all in the same cluster (all equal to 0)
    no_structure = rng.rand(n_samples, 2), np.zeros(n_samples)
    # ------------------------------------------------------------------------------------------------------------------

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    # ------------------------------------------------------------------------------------------------------------------
    #  Tai-Chi3 dataset: a dataset shaped like Ying-Yang with color values incorporated into the feature space

    def make_taichi3_dataset(n_samples, random_state, boundary_gap=0.2):
        rng = np.random.RandomState(random_state)

        # Sample points from the image
        points = []
        labels = []
        while len(points) < n_samples:
            x = rng.uniform(-2, 2)
            y = rng.uniform(-2, 2)
            z = x + 1j * y

            if np.abs(z) > 2 - boundary_gap:
                label = 1  # Gray
            elif np.abs(z) <= 2 - 2 * boundary_gap:
                if ((np.real(z) > 0) + (np.abs(z - 1j) < 1 - boundary_gap / 2)) * (
                        np.abs(z - 1j) > 0.3 + boundary_gap / 2) * (np.abs(z + 1j) > 1 + boundary_gap / 2):
                    label = 2  # White
                elif ((np.real(z) < 0) + (np.abs(z + 1j) < 1 - boundary_gap / 2)) * (
                        np.abs(z + 1j) > 0.3 + boundary_gap / 2) * (np.abs(z - 1j) > 1 + boundary_gap / 2):
                    label = 0  # Black
                elif np.abs(z + 1j) < 0.3 - boundary_gap / 2:
                    label = 2  # White
                elif np.abs(z - 1j) < 0.3 - boundary_gap / 2:
                    label = 0  # Black
                else:
                    continue
            else:
                continue

            points.append([x, y])
            labels.append(label)

        X = np.array(points)
        y = np.array(labels)

        return X, y

    X, y = make_taichi3_dataset(n_samples * 10, random_state=seed)

    taichi3 = (X, y)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def make_spiral3_dataset(n_samples, noise=0.01, random_state=None):
        rng = np.random.RandomState(random_state)
        t = np.linspace(0, 0.7, n_samples // 3)  # Reduced range to make spirals smaller
        dx1, dy1 = t * np.cos(4 * np.pi * t), t * np.sin(4 * np.pi * t)
        dx2, dy2 = t * np.cos(4 * np.pi * t + 2 * np.pi / 3), t * np.sin(4 * np.pi * t + 2 * np.pi / 3)
        dx3, dy3 = t * np.cos(4 * np.pi * t + 4 * np.pi / 3), t * np.sin(4 * np.pi * t + 4 * np.pi / 3)
        X = np.vstack((np.column_stack((dx1, dy1)),
                       np.column_stack((dx2, dy2)),
                       np.column_stack((dx3, dy3))))
        y = np.hstack((np.zeros(n_samples // 3),
                       np.ones(n_samples // 3),
                       np.full(n_samples // 3, 2)))
        X += noise * rng.randn(*X.shape)

        # Add a hole at the center
        center_mask = np.sum(X ** 2, axis=1) > 0.01  # Reduced hole size
        X = X[center_mask]
        y = y[center_mask]

        return X, y

    spiral3 = make_spiral3_dataset(n_samples=n_samples * 3, random_state=seed)  # Tripled n_samples for denser points
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Tai-Chi dataset: a dataset shaped like the Yin-Yang symbol
    # Define the grid with adjustable point density
    # Increase step size for lower density, decrease for higher density
    step_size = 0.01  # Adjust this value as needed
    x = np.arange(-2, 2 + step_size, step_size)
    y = np.arange(-2, 2 + step_size, step_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Initialize the Taichi matrix
    taichi = np.zeros_like(Z, dtype=float)

    # Outer circle (gray background)
    mask_outer = np.abs(Z) > 2
    taichi += 0.5 * mask_outer

    # Inner patterns
    mask_inner = np.abs(Z) <= 2
    # Adjust conditions to leave space between parts
    sum_cond = ((np.real(Z) > 0).astype(int) + (np.abs(Z - 1j) < 1).astype(int))
    mask2 = np.abs(Z - 1j) > 0.3  # Increase threshold to leave more space
    mask3 = np.abs(Z + 1j) > 1
    taichi += mask_inner * sum_cond * mask2 * mask3

    # Small circle at the bottom
    taichi += (np.abs(Z + 1j) < 0.3).astype(float)

    # Prepare the image
    TC = np.stack([taichi] * 3, axis=-1)  # Stack to create RGB channels
    TC = 255 * TC  # Scale values for display
    TC = TC.astype(np.uint8)  # Convert to unsigned 8-bit integer

    def make_taichi_dataset(n_samples, noise=0.01, random_state=None):
        rng = np.random.RandomState(random_state)

        # Sample points from the image
        points = []
        labels = []
        while len(points) < n_samples:
            x = rng.uniform(-2, 2)
            y = rng.uniform(-2, 2)
            pixel_x = int((x + 2) / 4 * TC.shape[1])
            pixel_y = int((y + 2) / 4 * TC.shape[0])
            if TC[pixel_y, pixel_x, 0] > 0:
                # Make the background (gray) sparser
                if x ** 2 + y ** 2 < 5 and TC[pixel_y, pixel_x, 0] > 10 and TC[pixel_y, pixel_x, 0] < 245:
                    continue
                if TC[pixel_y, pixel_x, 0] > 10 and TC[
                    pixel_y, pixel_x, 0] < 245 and rng.random() > 0.1:  # Only keep 30% of gray points
                    continue
                points.append([x, y])
                labels.append(TC[pixel_y, pixel_x, 0] // 128)  # 0 for black, 1 for gray, 2 for white

        X = np.array(points)
        y = np.array(labels)

        # Add noise
        X += noise * rng.randn(*X.shape)

        return X, y

    X, y = make_taichi_dataset(n_samples * 10, random_state=seed)

    taichi = (X, y)
    # ------------------------------------------------------------------------------------------------------------------

    return [
        ("taichi3", taichi3),
        ("taichi", taichi),
        ("spiral3", spiral3),
        ("noisy_circles", noisy_circles),
        ("noisy_moons", noisy_moons),
        ("varied", varied),
        ("aniso", aniso),
        ("blobs", blobs),
        ("no_structure", no_structure),
    ]


class LeidenClustering:
    def __init__(self, A, resolution_parameter=1.0):
        self.A = A
        self.resolution_parameter = resolution_parameter
        self.labels_ = None
        self.modularity_score_ = None

    def fit(self, X):
        import igraph as ig
        import leidenalg as la
        # Convert the sparse matrix to an igraph graph
        sources, targets = self.A.nonzero()
        weights = self.A.data
        edges = list(zip(sources, targets))
        g = ig.Graph(edges=edges, directed=True)
        g.es['weight'] = weights.tolist()
        # Run the Leiden algorithm for community detection
        partition = la.find_partition(g, la.RBConfigurationVertexPartition,
                                      resolution_parameter=self.resolution_parameter)
        # Extract the membership vector and modularity score
        self.labels_ = np.array(partition.membership)
        self.modularity_score_ = partition.modularity
        return self