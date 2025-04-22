function A = grapheneSheet( m, n ) 
% 
%  A = grapheneSheet( m, n ) ; 
% 
%  generates the adjacency matrix of 
%  a graphSheet with lattice of m rows and n columns 
% 
%  The matrix is in sparse structure 
% 


% ... construct the adajency matrix (using Kronecker products)  

% ...  connect nodes on vertical lines  

I = eye(n);
J = diag(ones(m-1,1),1);   % super-diagonal of block size m  
A = kron(J,I);             % n copies 

A = sparse(A) ; 

% ... connect nodes in horizontal lines  of odd indices 

d1 = zeros(n-1,1);
d1(1:2:n-1) = 1;

d2 = zeros(m,1);
d2(1:2:m) = 1;

A = A + sparse( kron(diag(d2),diag(d1,1)) ) ; 

% ... connect nodes in horizontal lines of eve incides 

d1 = zeros(n-1,1);
d1(2:2:n-1) = 1;

d2 = zeros(m,1);
d2(2:2:m) = 1;

A = A + sparse( kron(diag(d2),diag(d1,1)) ); 

Nodes = size(A,1);

% ... symmetrize the ajacency matrix 

A = A + A'; 

return 


% --------------------------------------------------------
% Nikos P. Pitsianis, nikos@cs.duke.edu, 
% on building the Adjacency matrix 
% 
% Xiaobai Sun, xiaobai@cs.duke.edu 
% on spectral embedding and visual display 
% 
% January 10, 2012 
% 
% Contact xiaobai or nikos for using the functions 
% and material for purpose other than the homework 
% --------------------------------------------------------
