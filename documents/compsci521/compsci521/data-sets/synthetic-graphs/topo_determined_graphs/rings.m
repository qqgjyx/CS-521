function [A, gname, L, X] = rings(n,k)
  % A = rings(n,k); 
  % generates the adjancecy of an undirected circular graph 
  % with n nodes and semi-bandwidth k >= 1 
  % in sparse format 
  % 
  
  if nargin == 1 
      k = 1; 
  end
  
  J = eye(n);
  J = J(:,[n,1:n-1]);
  A = J; 
  for j = 1:k-1 
      A = J + A*J; 
  end
  
  A = A + A';
  A = sparse(A); 
  L = ones(n, 1);
  X = [];

  gname = sprintf('rings(%d,%d)',n,k);
  
end
  
%% Programmer 
%% Xiaobai Sun 
%% 