# Juntang
# 2024/09/12
# loader.py

import numpy as np
from scipy.sparse import coo_matrix, spdiags

# Base Graph class (handles undirected graphs)
class Graph:
    def __init__(self, path, skrows=0):
        self.path = path
        self.edge_list = np.loadtxt(path, delimiter='\t', skiprows=skrows)
        self.X = self.edge_list[:, :-1]
        self.y = self.edge_list[:, -1]
        self._B = None  # Incidence matrix placeholder
        self._L = None  # Laplacian matrix placeholder
        self._D = None  # Degree matrix placeholder
        self._A = None  # Adjacency matrix placeholder

        # Step 1: Remove duplicate edges and self-loops
        # self.edge_list = self.remove_duplicate_edges(self.edge_list)
        # self.edge_list = self.remove_self_loops(self.edge_list)

        # Compute m and n
        self.m = self.edge_list.shape[0]  # Number of edges
        self.nodes = np.unique(self.edge_list[:, :2])  # Unique nodes
        self.n = len(self.nodes)  # Number of nodes

    def remove_duplicate_edges(self, edge_list):
        """Remove duplicate edges only for undirected graphs, keep for directed graphs."""
        if isinstance(self, Graph):  # Only apply to undirected graphs
            # Sort each edge so (a, b) is the same as (b, a) for undirected graphs
            sorted_edges = np.sort(edge_list[:, :2], axis=1)

            # Find unique edges and identify duplicates
            unique_edges, indices = np.unique(sorted_edges, axis=0, return_index=True)
            duplicates = np.setdiff1d(np.arange(sorted_edges.shape[0]), indices)

            if duplicates.size > 0:
                print(f"Found {duplicates.size} duplicate edges. Here are the first 5 duplicates:")
                print(sorted_edges[duplicates[:5]])
            else:
                print("No duplicate edges found.")

            return unique_edges
        else:
            # For directed graphs, keep all edges
            return edge_list

    def remove_self_loops(self, edge_list):
        """Remove self-loops from the edge list."""
        self_loops = edge_list[edge_list[:, 0] == edge_list[:, 1]]
        if len(self_loops) > 0:
            print(f"Found {len(self_loops)} self-loops. Here are the first 5 self-loops:")
            print(self_loops[:5])
        else:
            print("No self-loops found.")
        return edge_list[edge_list[:, 0] != edge_list[:, 1]]  # Return edges without self-loops

    def B(self):
        """Construct and return the incidence matrix B."""
        if self._B is not None:
            return self._B

        # Map original node indices to consecutive indices
        node_map = {node: i for i, node in enumerate(self.nodes)}

        rows = np.concatenate([self.edge_list[:, 0], self.edge_list[:, 1]])
        rows = np.array([node_map[node] for node in rows])
        cols = np.concatenate([np.arange(self.m), np.arange(self.m)])
        vals = np.concatenate([np.ones(self.m), -np.ones(self.m)])

        self._B = coo_matrix((vals, (rows, cols)), shape=(self.n, self.m)).tocsr()
        return self._B

    def L(self):
        """Compute and return the Laplacian matrix L."""
        if self._L is not None:
            return self._L
        B = self.B()
        self._L = B @ B.T  # L = B * B'
        return self._L

    def D(self):
        """Compute and return the Degree matrix D."""
        if self._D is not None:
            return self._D
        L = self.L()
        diag_L = L.diagonal()
        self._D = spdiags(diag_L, 0, L.shape[0], L.shape[1])
        return self._D

    def A(self):
        """Compute and return the Adjacency matrix A."""
        if self._A is not None:
            return self._A
        D = self.D()
        L = self.L()
        self._A = D - L
        return self._A

    def assert_adjacency_matrix(self):
        """Assert properties of the adjacency matrix A for undirected graphs."""
        A = self.A()

        # Assertion 1: All entries are non-negative
        assert A.min() >= 0, "The adjacency matrix contains negative entries."

        # Assertion 2: No self-loops
        diag = A.diagonal()
        assert np.all(diag == 0), "The adjacency matrix contains non-zero diagonal entries (self-loops)."

        # Assertion 3: Graph is connected
        row_sums = A.sum(axis=1)
        zero_row_indices = np.where(row_sums == 0)[0]
        if len(zero_row_indices) > 0:
            print(f"Zero row(s) detected at indices: {zero_row_indices}")
            raise ValueError(f"The adjacency matrix contains zero row(s). Graph is disconnected.")

        # Assertion 4: Correct dimensions
        assert A.shape == (self.n, self.n), f"Adjacency matrix shape mismatch: expected ({self.n}, {self.n}), got {A.shape}"

        # Assertion 5: Number of edges matches
        num_non_zero_entries = A.nnz
        expected_non_zero_entries = 2 * self.m
        assert num_non_zero_entries == expected_non_zero_entries, \
            f"Number of non-zero entries mismatch: expected {expected_non_zero_entries}, got {num_non_zero_entries}"
        print("All assertions passed.")

# Subclass DiGraph for directed graphs
class DiGraph(Graph):
    def B(self):
        """Construct and return the incidence matrix B for directed graphs."""
        if self._B is not None:
            return self._B

        node_map = {node: i for i, node in enumerate(self.nodes)}

        # +1 for the source node, -1 for the target node of each directed edge
        rows = np.concatenate([self.edge_list[:, 0], self.edge_list[:, 1]])
        rows = np.array([node_map[node] for node in rows])
        cols = np.concatenate([np.arange(self.m), np.arange(self.m)])
        vals = np.concatenate([np.ones(self.m), -np.ones(self.m)])

        self._B = coo_matrix((vals, (rows, cols)), shape=(self.n, self.m)).tocsr()
        return self._B

    def A(self):
        """Compute and return the Adjacency matrix A for directed graphs."""
        if self._A is not None:
            return self._A

        # Construct adjacency matrix directly from the edge list
        node_map = {node: i for i, node in enumerate(self.nodes)}
        rows = np.array([node_map[node] for node in self.edge_list[:, 0]])
        cols = np.array([node_map[node] for node in self.edge_list[:, 1]])
        vals = np.ones(self.m)

        self._A = coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()
        return self._A

    def assert_adjacency_matrix(self):
        """Assert properties of the adjacency matrix A for directed graphs."""
        A = self.A()

        # Assertion 1: Non-negative entries
        assert A.min() >= 0, "The adjacency matrix contains negative entries."

        # # Assertion 2: No self-loops
        # diag = A.diagonal()
        # assert np.all(diag == 0), "The adjacency matrix contains non-zero diagonal entries (self-loops)."

        # Assertion 3: Correct dimensions
        assert A.shape == (self.n, self.n), f"Adjacency matrix shape mismatch: expected ({self.n}, {self.n}), got {A.shape}"

        print("All assertions passed for directed graph.")