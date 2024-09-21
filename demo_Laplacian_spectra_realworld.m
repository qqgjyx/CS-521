function Leig = demo_Laplacian_spectra_realworld(A, gname, bnormalized)
% Function: demo_Laplacian_spectra_realworld
% This function computes the Laplacian spectrum and performs spectral
% embedding in 2D and 3D for a given adjacency matrix.
%
% Inputs:
%   A          - Adjacency matrix of the graph (nxn matrix)
%   gname      - Graph name to be used in plots
%   bnormalized - Logical flag for normalized Laplacian (1 for normalized)
%
% Output:
%   Leig       - Structure containing eigenvalues and eigenvectors of the Laplacian
%
% Example usage:
%   Leig = demo_Laplacian_spectra_realworld(A, 'Facebook', 1);

    fprintf('\n   %s begin \n\n', mfilename ); 

    % Check if inputs are provided
    if nargin < 3
        error('Adjacency matrix A, graph name, and normalization flag must be provided.');
    end

    n = size(A, 1);  % Number of nodes
    Leig = struct();  % Initialize output structure

    % Compute Laplacian matrix
    d = sum(A, 2);  % Degree vector
    if bnormalized
        % Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        dh = sqrt(d);  % Square root of degrees
        L = eye(n) - diag(1 ./ dh) * A * diag(1 ./ dh);
    else
        % Unnormalized Laplacian: L = D - A
        L = diag(d) - A;
    end

    % Determine whether to use full or sparse solver based on graph size
    if n < 3500
        fprintf('Computing full Laplacian spectrum...\n');
        [V, S] = eig(full(L));  % Full eigenvalue decomposition
        S = diag(S);  % Convert eigenvalue matrix to vector
    else
        % Sparse solver for larger graphs (computing only a few eigenvalues)
        sigma = 1e-4;  % Shifted eigenvalue parameter
        k = min(4000, n-1);  % Number of eigenvalues to compute
        opts.isreal = true;
        opts.issym = true;
        
        fprintf('Using sparse solver with sigma = %g for better stability...\n', sigma);
        [V, S] = eigs(L, k, sigma, opts);  % Shifted eigenvalue decomposition
        S = diag(S);  % Convert eigenvalue matrix to vector
    end

    % Sort eigenvalues in ascending order and rearrange eigenvectors accordingly
    [S, idx] = sort(S, 'ascend');
    V = V(:, idx);  % Reorder eigenvectors

    % Store eigenvalues and eigenvectors in Leig structure
    Leig.S = S;  % Sorted eigenvalues
    Leig.V = V;  % Corresponding eigenvectors

    % Plot the eigenvalue spectrum
    figure;
    plot(S, 'bx');
    title([gname, ' Laplacian eigenvalues']);
    xlabel('Eigen-mode index'); 
    ylabel('Eigenvalue');

    % Plot the eigenvector matrix as a heatmap
    figure;
    imagesc(V);  % V contains the eigenvectors
    colorbar;    % Display color scale
    caxis([-0.1, 0.1]);  % Adjust the range to capture small variations
    axis image;
    title([gname, ' Laplacian Eigenvectors']);
    xlabel('Eigen-mode index');
    ylabel('Eigenvector/column');

    % Perform spectral embedding based on low-energy modes (first few eigenvectors)
    num_eigenvectors = size(V, 2);
    xyzidx = min([2, 3, 4], num_eigenvectors);  % Ensure we don't exceed available eigenvectors

    if num_eigenvectors >= 2
        fprintf('2D vertex embedding with low-energy modes...\n');
        figure;
        plot(graph(A), 'XData', V(:, xyzidx(1)), 'YData', V(:, xyzidx(2)));
        axis equal; box on; grid on;
        xlabel(sprintf('Mode %d', xyzidx(1)));
        ylabel(sprintf('Mode %d', xyzidx(2)));
        title([gname, ' Laplacian Embedding 2D']);
    end

    if num_eigenvectors >= 3
        fprintf('3D vertex embedding with low-energy modes...\n');
        figure;
        plot(graph(A), 'XData', V(:, xyzidx(1)), 'YData', V(:, xyzidx(2)), 'ZData', V(:, xyzidx(3)));
        axis equal; box on; grid on; rotate3d;
        xlabel(sprintf('Mode %d', xyzidx(1)));
        ylabel(sprintf('Mode %d', xyzidx(2)));
        zlabel(sprintf('Mode %d', xyzidx(3)));
        title([gname, ' Laplacian Embedding 3D']);
    end

    % Perform spectral embedding with high-energy modes (last few eigenvectors)
    if num_eigenvectors >= 3
        xyzidx_high = num_eigenvectors - (0:2);  % Highest eigenmodes
        fprintf('3D vertex embedding with high-energy modes...\n');
        figure;
        plot(graph(A), 'XData', V(:, xyzidx_high(1)), 'YData', V(:, xyzidx_high(2)), 'ZData', V(:, xyzidx_high(3)));
        axis equal; box on; grid on; rotate3d;
        xlabel(sprintf('Mode %d', xyzidx_high(1)));
        ylabel(sprintf('Mode %d', xyzidx_high(2)));
        zlabel(sprintf('Mode %d', xyzidx_high(3)));
        title([gname, ' Laplacian Embedding 3D (High-energy modes)']);
    end

    fprintf('\n\n   %s end \n\n', mfilename);
end