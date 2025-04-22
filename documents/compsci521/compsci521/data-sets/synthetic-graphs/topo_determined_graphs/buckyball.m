function [A,str,L,X] = buckyball(opts)

arguments
    opts.symmetrize (1,1) logical = false
    opts.unweighted (1,1) logical = false
    opts.dropself (1,1) logical = true
  end

  A = bucky();

  % make sure we have no diagonal entries (self-loops)
  if opts.symmetrize
    A = A + A';
  end

  if opts.unweighted
    A = double( A > 0 );
  end

  if opts.dropself
    A = A - diag(diag(A));
  end



  str = 'buckyball';
  L = ones(size(A,1),1);
  X = [];
end