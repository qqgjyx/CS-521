function [A, gname] = wheel(n )
%
%  A = wheel(n);
% 
% generates the adjacency matrix of a wheel granph with n+1  nodes 
% 


J = eye(n+1);
J(1,1) = 0; 
J = J(:, [n+1, 1:n]);
A = zeros(n+1,n+1);
A(1,:)   = 1; 
A(2,n+1) = 1; 
A = A + J; 
A = (A + A') > 0 ; 

gname = sprintf('wheel(%d)',n);

return 

% =======================
% Programmer: 
% Xiaobai Sun 
% Date: June 21, 2024 