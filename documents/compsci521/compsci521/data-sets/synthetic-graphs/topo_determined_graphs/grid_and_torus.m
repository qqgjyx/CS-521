function [A,str,L,X] = grid_and_torus(d, ns, btorus)
% 
% function A = grid_and_torus(d, nd, btorus);
% 
%   generates the adjacency matrix of 
%   a d-dimensional grid or torus or in between 
% 
%  INPUT: 
%  d:  integer, d>=1, the number of dimensions 
%  ns: integer or integer array 
%      ns(j) = the length of dimension-j 
%      if all the same, ns is a single integer 
%  btorus: logic-vale, or logic value array, 
%          btosus(j) = circular wrapping in dimension-j 
%          if all the same, btorus is a single logic value 
%          defaut: btorus = 0 
%  OUTPUT:
%  A : sparse binary valued array, 
%       for the adjacency matrix of a grid or torus 
%  
% ----------------------------
% Programmer: Nikos Pitsianis 
% 2022 
% ------------------------------

if nargin == 2
  btorus = 0;      % default setting, no torus 
end

if isscalar( btorus )
  btorus = btorus * ones(d,1);
end

if isscalar(ns)
  ns= ns * ones(d,1);
end

assert(d == length(ns) && d == length(btorus))

A = sparse( prod(ns), prod(ns) );
for i = 1:d
  J = spdiags(ones(ns(i)-1,1),-1,sparse(ns(i),ns(i)));
  J = J + J';

  M2 = J;
  if ns(i) > 2
    M2(ns(i),1) = btorus(i);
    M2(1,ns(i)) = btorus(i);
  end
  
  M1 = speye( prod(ns(i+1:d)) );
  M3 = speye( prod(ns(1:i-1)) );
  A  = A + kron(M1,kron(M2,M3));
end

str = sprintf('grid(%d%s)', ns(1), sprintf(',%d',ns(2:end)) );

L = ones(size(A,1),1);
X = [];

end

%% Created by Nikos Pitsianis 
%% document revised by Xiaobai Sun 
%% 