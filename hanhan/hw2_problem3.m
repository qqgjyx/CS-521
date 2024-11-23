% Problem 3: Empirical Analysis of a Digraph Representing a Real-World Network
close all;
clear all;

% Load dataset for digraph analysis (choose Iris dataset for demonstration)
get_iris_data;
X = meas;

% Parameters for constructing a k-NN directed graph
k = 5;  % Number of nearest neighbors
D = pdist2(X, X);  % Compute pairwise distances
A_knn = zeros(size(D));  % Adjacency matrix for directed k-NN graph

% Construct the directed k-NN graph by connecting each node to its k-nearest neighbors
[~, idx] = sort(D, 2);  % Sort distances for each node
for i = 1:size(D, 1)
    A_knn(i, idx(i, 2:k+1)) = 1;  % Only connect to k nearest neighbors
end

% Convert to digraph and analyze connectivity
G = digraph(A_knn);  % Directed graph
components = conncomp(G, 'Type', 'weak');  % Find weakly connected components
num_components = max(components);
disp(['Number of weakly connected components: ', num2str(num_components)]);

% Identify the largest connected component (LCC)
LCC_nodes = mode(components);
LCC = subgraph(G, find(components == LCC_nodes));  % Subgraph of LCC

% Check if the LCC is strongly connected
is_strongly_connected = all(conncomp(LCC, 'Type', 'strong') == 1);
disp(['LCC is strongly connected: ', num2str(is_strongly_connected)]);

% Part (b) - Perron Distribution xp for the LCC

% Parameters for variational BP approach
alpha_values = linspace(0.85, 1, 4);  % Alpha values between 0.85 and 1
num_b = 5;  % Number of probing vectors
n_LCC = numnodes(LCC);  % Number of nodes in LCC

% Generate probing vectors
b_vectors = rand(n_LCC, num_b);
b_vectors = b_vectors ./ sum(b_vectors);  % Normalize to make sum of each column 1

% Compute Perron distribution xp for each alpha and b
xp_results = zeros(n_LCC, length(alpha_values), num_b);

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    A_alpha = alpha * adjacency(LCC)' + (1 - alpha) * (ones(n_LCC) / n_LCC);  % Adjusted adjacency

    for j = 1:num_b
        b = b_vectors(:, j);
        [xp, ~] = eigs(A_alpha, 1, 'largestreal');  % Compute Perron vector
        xp_results(:, i, j) = xp / sum(xp);  % Normalize xp
    end
end

% Part (c) - Plotting xp variation with alpha and b
figure;
for j = 1:num_b
    subplot(ceil(num_b/2), 2, j);  % Arrange subplots
    plot(alpha_values, squeeze(xp_results(:, :, j))');
    xlabel('\alpha');
    ylabel('Perron Distribution (xp)');
    title(['Perron Distribution for probing vector b_', num2str(j)]);
    legend(arrayfun(@(x) ['Node ', num2str(x)], 1:n_LCC, 'UniformOutput', false));
end
sgtitle('Perron Distribution xp Variation with \alpha and Probing Vectors b');
