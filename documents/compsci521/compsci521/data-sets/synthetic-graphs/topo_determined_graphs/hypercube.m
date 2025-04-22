function [A,str,L,X] = hypercube(d)
  % A = hypercube(d); 
  % 
  % Generate the adjacency matrix of a hypercube of d dimensions
  % 
  % 
  
  A = subcube( int64(d) );  % recursion function 

  % extra information not needed in recursion 
  L = ones(size(A,1), 1);
  X = [];
  str = sprintf('%d-cube',d);

end

function A = subcube(d)
if d == 0
    A = 0;
  else
    Ap = subcube(d-1);
  
    I = speye(2^(d-1));
  
    A = [Ap I; I Ap];
end
end

%%% ==================== 
%%  created by Nikos Pitsianis 
%% 