function A = knn2adjacency( idx, dst ) 
% 
%   SYNTAX
%
%   A = knn2adjacency( IDX, DST ); 
% 
%  knn2adjacency - converts the knnsearch results to a 
%                  knn-graph with adjacency matrix A in sparse format 
% 
% INPUT
%
%   IDX  nxk integer array 
%            IDX(i,j) is the index to the j-th nearest neighbor of point i 
%   DST  nxk real-valued, non-negative array 
%            DST(i,j) is the distance between i and and its j-th neighbor 
%
% OUTPUT
% 
%   A    nxn Adjacy matrix 
%        column  A(:,i) contains the k-neighbors of source point i 
%         A( i,j ) is nonzero if i is in DST( j, :)   
%
%        row A(i,:) constains all souce points with $i$ as a neighbor 
% 
% EXAMPLE
%
%    [idx,dst] = knnsearch(X,X,'k',k);       % all-to-all kNN graph 
%    A         = knn2adjacency( idx, dst);
%
% DEPENDENCIES
%
%   <none>
%
%
% See KNNSEARCH 
%
  
  % n = #feature-points, k = #neighbors 
  [n, k] = size( idx );
  
  % column indices (for sparse matrix formation convenience)
  idxCol = repmat( (1 : n).', [1 k] );
  
  % sparse matrix formation
  A = sparse( idx, idxCol, dst, n, n );
  
end



%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 30, 2017
%
% CHANGELOG
%
%   0.1 (Dec 30, 2017) - Dimitris
%       * initial implementation
%   document revision: Oct. 20, 2024 
%                      Nikos Pitsianis 
%                      Xiaobai Sun 
% ------------------------------------------------------------

