function [A,str,L,X] = star(n)

  A = [0 ones(1,n); ones(n,1) zeros(n,n)];
  L = ones(n+1, 1);
  X = [];
  str = sprintf('star(%d)',n);

end