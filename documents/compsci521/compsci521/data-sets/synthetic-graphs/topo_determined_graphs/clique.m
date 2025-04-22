function [A,str,L,X] = clique(n, edgedrop, seed)

  arguments
    n (1,1) {mustBeInteger,mustBePositive} = 10
    edgedrop (1,1) = 0.0
    seed (1,1) {mustBeInteger} = 0
  end

  A = sparse( ones(n, n) - eye(n) );

  if edgedrop > 0
    [i,j] = find(triu(A));
    idx = randperm(numel(i), max( round(numel(i)*edgedrop), 1 ) );
    idx = setdiff(1:numel(i), idx);
    A = sparse(i(idx), j(idx), 1, n, n);
    A = A + A';
  end

  L = ones(n, 1);
  X = [];

  str = sprintf('clique(%d)', n);

end