function mat = rnn2adjacency(idxCol, dist)
% INPUT         idxCol  Index columns for matrix        [n-cell]
%               dist    Distances of points             [n-cell]
% OUTPUT        mat     Sparse matrix with distances    [n-by-n sparse]

% number of neighbors for each point
nNbr = cellfun( @(x) numel(x), idxCol );

% number of points
n = numel( idxCol );

% row indices (for sparse matrix formation convenience)
idxRow = arrayfun( @(n,i) i * ones( 1, n ), nNbr, (1:n)', ...
                   'UniformOutput', false );

% sparse matrix formation by MATLAB convension 
mat = sparse( [idxRow{:}], [idxCol{:}], [dist{:}], n, n );

end

%% Programmer 
%% Dimitris Floros 
%% Last revision 