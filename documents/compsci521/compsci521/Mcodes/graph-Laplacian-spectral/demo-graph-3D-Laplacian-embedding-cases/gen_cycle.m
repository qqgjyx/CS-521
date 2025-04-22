function A = gen_cycle( n, figID) 
% 
% A = genDcube( n, figID) ;   % d >= 1
% 
% generate  the adjacency matrix A for the cycle of length n 
% 
% when nargin > 1, display the generation process 
% 

if n < 3 
    error('a cycle contains at least 3 nodes' );
end

A = eye(n);
A = A(:, [2:n,1] );
A = A + A';

if nargin > 1 
    figure( figID )
    spy(A) 
    title(sprintf('adjacency matrix of cycle graph of %d nodes', n) ); 
end

return 

%%  ---------------------------------------------------
% Xiaobai Sun 
% Duke CS 
% For Numerical Analysis Class
% Oct. 2011 
%  
% Contact xiaobai for any further distribution 
% ---------------------------------------------------
