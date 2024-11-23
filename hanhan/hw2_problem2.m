% Problem 2 Solution: Vertex-to-Vector Encoding and Embedding
close all;
clear all;

% Load Iris dataset using get_iris_data function
get_iris_data;  % Loads 'meas' as X and Lcolors for visualization
X = meas;       % Feature data from Iris dataset

% Part (a): Construct two graphs - one weighted and one unweighted

% Unweighted k-NN graph (k = 5)
k = 5;  % Number of neighbors
D_unweighted = pdist2(X, X);  % Compute pairwise distances
A_unweighted = zeros(size(D_unweighted));
[~, idx_unweighted] = sort(D_unweighted, 2);
for i = 1:size(D_unweighted, 1)
    A_unweighted(i, idx_unweighted(i, 2:k+1)) = 1;  % k nearest neighbors
end
A_unweighted = max(A_unweighted, A_unweighted');  % Make symmetric

% Weighted graph with Gaussian similarity (sigma = 1)
sigma = 1.0;
A_weighted = exp(-D_unweighted.^2 / (2 * sigma^2));  % Gaussian similarity
A_weighted(A_weighted < 0.1) = 0;  % Sparsify graph by thresholding

% Part (b): Spectral embedding using the normalized Laplacian, d > 2
d = 3;  % Embedding dimension
sigma_eigs = 1e-5;  % Small shift for stability in eigs function

% Unweighted graph embedding
D_unweighted_deg = diag(sum(A_unweighted, 2));
L_unweighted = D_unweighted_deg - A_unweighted;
L_unweighted_norm = D_unweighted_deg^(-1/2) * L_unweighted * D_unweighted_deg^(-1/2);
[V_unweighted, ~] = eigs(L_unweighted_norm, d + 1, sigma_eigs);
spectral_embedding_unweighted = V_unweighted(:, 2:end);

% Weighted graph embedding
D_weighted_deg = diag(sum(A_weighted, 2));
L_weighted = D_weighted_deg - A_weighted;
L_weighted_norm = D_weighted_deg^(-1/2) * L_weighted * D_weighted_deg^(-1/2);
[V_weighted, ~] = eigs(L_weighted_norm, d + 1, sigma_eigs);
spectral_embedding_weighted = V_weighted(:, 2:end);

% Part (c): Difference in Pairwise Distances for Weighted Graph
pairwise_distances_original = D_unweighted;
pairwise_distances_embedding = pdist2(spectral_embedding_weighted, spectral_embedding_weighted);
distance_diff = abs(pairwise_distances_original - pairwise_distances_embedding);

figure;
imagesc(distance_diff);
colorbar;
title('Difference in Pairwise Distances for Weighted Graph');

% Part (d): Community Detection using Fiedler Vector

% Unweighted graph - Fiedler vector-based community detection
fiedler_vector_unweighted = V_unweighted(:, 2);
community_labels_unweighted = fiedler_vector_unweighted > 0;
community_colors_unweighted = community_labels_unweighted + 1;

% Weighted graph - Fiedler vector-based community detection
fiedler_vector_weighted = V_weighted(:, 2);
community_labels_weighted = fiedler_vector_weighted > 0;
community_colors_weighted = community_labels_weighted + 1;

% 3D scatter plot of communities for weighted graph
figure;
scatter3(spectral_embedding_weighted(:, 1), spectral_embedding_weighted(:, 2), spectral_embedding_weighted(:, 3), ...
    20, community_colors_weighted, 'filled');
title('Community Detection in Weighted Graph using Fiedler Vector');
xlabel('Embedding Dimension 1');
ylabel('Embedding Dimension 2');
zlabel('Embedding Dimension 3');

% Highlight cut edges for weighted graph
G_weighted = graph(A_weighted);
cut_edges = find(community_labels_weighted ~= community_labels_weighted');
hold on;
for e = cut_edges'
    [u, v] = ind2sub(size(A_weighted), e);
    plot3([spectral_embedding_weighted(u, 1), spectral_embedding_weighted(v, 1)], ...
          [spectral_embedding_weighted(u, 2), spectral_embedding_weighted(v, 2)], ...
          [spectral_embedding_weighted(u, 3), spectral_embedding_weighted(v, 3)], 'k--');
end
hold off;

% Part (d) 2D plot for unweighted graph
figure;
scatter3(spectral_embedding_unweighted(:, 1), spectral_embedding_unweighted(:, 2), spectral_embedding_unweighted(:, 3), ...
    20, community_colors_unweighted, 'filled');
title('Community Detection in Unweighted Graph using Fiedler Vector');
xlabel('Embedding Dimension 1');
ylabel('Embedding Dimension 2');
zlabel('Embedding Dimension 3');
