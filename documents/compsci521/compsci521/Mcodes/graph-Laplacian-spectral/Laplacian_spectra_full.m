function Leig  = Laplacian_spectra_full( A, bnormalized ) 
% 
% Leig = Laplacian_spectra_full( A, bnormalized ) ;
%  
% INPUT 
%  A : nxn array, for a symmetric matrix, A' = A 
%  bnormalzied: logic 1 --> normalized Laplacian 
% 
% OUTPUT 
% Leig.normalized = bnormalized 
% Leig.U: nxn array, for the full eigenvector matrix 
% Leig.S: nx1 array, for the eigenvalues, in non-descending order 
% Leig.kCCs = number of Connected Components 

% --------------------------------------------

if ~issymmetric(A) 
    error('the matrix is not symmetric');
end

d = sum(A,1);         % d(i) = the degree of node i 
if min(d) == 0       
   error('there are isolated nodes'); 
end

n = size(A,1);        % the number of nodes 
m = nnz(A)/2;         % the number of edges 

% ... form the Laplacian matrix explicitly 
Leig.normalzed = bnormalized;
if bnormalized 
   dh = sqrt(d);
   L  = eye(n) - diag(1./dh) * A * diag(1./dh); 
else 
   L = diag(d) - A ;    
end 

% ... get the Laplacian spectral 

L = full(L);                   % in order to use the dense eigen solver

[ V, S] = eig(L);  
S       = diag(S);             % the eigenvalues into a vector  
[S, p]  = sort(S, 'ascend' );  % in non-descreasing order 
Leig.S  = S; 
Leig.V  = V(:, p);             % permuate the eigenvectors accordingly 

Leig.kCCs = sum( find(S < 1.5*eps ));  % number of numerically zero eigenvalues  

return

% ---------------------------------------
% Xiaobai Sun, Duke CS 
% ----------------------------------------

