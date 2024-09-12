select_topological_graphs;

d = sum(A, 2);
plot_distribution(d, 'Degree Sequence', 20, 1);

G = graph(A);
figure
plot(G)