% Problem 4: Differential Description of a Graph Sequence
close all;
clear all;

% Define the number of graphs in the sequence
q = 3;  % Adjust as needed (e.g., q = 3 for three graphs in sequence)

% Parameters for BA and WS models
num_nodes = 50;  % Number of nodes in each graph (adjustable based on dataset)
avg_degree = 4;  % Average degree for WS model and initial edges for BA model
rewiring_prob = 0.2;  % Rewiring probability for WS model

% Create global reference graph by merging all V and E
G_global = graph();  % Empty graph for global reference

% Initialize array to store individual graphs
G_seq = cell(1, q);

% Generate each Gi in sequence with some overlap in nodes
for i = 1:q
    % Alternate between custom BA and WS models
    if mod(i, 2) == 1
        % Create a BA graph with growth property
        G_seq{i} = createBAGraph(num_nodes, avg_degree);  % BA model
    else
        % Create a WS graph with rewiring property
        G_seq{i} = createWSGraph(num_nodes, avg_degree, rewiring_prob);  % Custom WS model
    end
    
    % Add the nodes and edges of G_i to the global reference graph
    G_global = addedge(G_global, G_seq{i}.Edges.EndNodes(:, 1), G_seq{i}.Edges.EndNodes(:, 2));
end

% Part (b) - Embedding the global reference graph in 2D or 3D space
X = rand(numnodes(G_global), 3);  % Random 3D positions for the global graph

% Plot the global reference graph in 3D
figure;
plot(G_global, 'XData', X(:,1), 'YData', X(:,2), 'ZData', X(:,3));
title('Global Reference Graph');
xlabel('X');
ylabel('Y');
zlabel('Z');

% Part (c) - Visualize Gi and Gi+1 on the global spatial reference map
figure;
for i = 1:q-1
    subplot(1, q-1, i);
    % Plot Gi in red and Gi+1 in blue with overlapping nodes highlighted
    Gi_nodes = unique(G_seq{i}.Edges.EndNodes);
    Gi_plus1_nodes = unique(G_seq{i+1}.Edges.EndNodes);
    
    hold on;
    plot(G_seq{i}, 'XData', X(Gi_nodes,1), 'YData', X(Gi_nodes,2), 'ZData', X(Gi_nodes,3), 'NodeColor', 'r');
    plot(G_seq{i+1}, 'XData', X(Gi_plus1_nodes,1), 'YData', X(Gi_plus1_nodes,2), 'ZData', X(Gi_plus1_nodes,3), 'NodeColor', 'b');
    
    % Highlight overlapping nodes
    overlap_nodes = intersect(Gi_nodes, Gi_plus1_nodes);
    scatter3(X(overlap_nodes,1), X(overlap_nodes,2), X(overlap_nodes,3), 50, 'k', 'filled');
    
    title(['Graphs G_{', num2str(i), '} and G_{', num2str(i+1), '}']);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    hold off;
end
sgtitle('Graph Sequence Visualization with Overlaps');

% Custom function to generate a BA graph
function G = createBAGraph(n, m)
    % Create a BA graph with 'n' nodes, each new node attaches to 'm' existing nodes
    if m >= n
        error('The number of edges to attach must be less than the total number of nodes.');
    end
    
    % Start with a fully connected initial network of m + 1 nodes
    G = graph();
    G = addnode(G, m+1);
    for i = 1:m+1
        for j = i+1:m+1
            G = addedge(G, i, j);
        end
    end
    
    % Add each new node and attach it to 'm' existing nodes based on degree
    for new_node = m+2:n
        G = addnode(G, 1);  % Add the new node
        degrees = degree(G);  % Get current degree of each node
        existing_nodes = 1:numnodes(G)-1;
        attach_nodes = datasample(existing_nodes, m, 'Weights', degrees(existing_nodes), 'Replace', false);
        for attach_node = attach_nodes
            G = addedge(G, new_node, attach_node);
        end
    end
end

% Custom function to generate a WS graph
function G = createWSGraph(n, k, beta)
    % Create a WS graph with 'n' nodes, each connected to 'k' nearest neighbors
    % Rewiring probability is 'beta'
    if mod(k, 2) ~= 0
        error('k must be even for the Watts-Strogatz model.');
    end
    
    % Initialize ring lattice
    G = graph();
    G = addnode(G, n);
    for i = 1:n
        for j = 1:k/2
            neighbor = mod(i + j - 1, n) + 1;  % Wrap around
            G = addedge(G, i, neighbor);
        end
    end
    
    % Rewire edges with probability beta
    for i = 1:n
        for j = 1:k/2
            if rand < beta
                % Rewire to a new target node
                G = rmedge(G, i, mod(i + j - 1, n) + 1);
                new_neighbor = randi(n);
                while new_neighbor == i || ismember(new_neighbor, neighbors(G, i))
                    new_neighbor = randi(n);  % Ensure no self-loops or duplicates
                end
                G = addedge(G, i, new_neighbor);
            end
        end
    end
end
