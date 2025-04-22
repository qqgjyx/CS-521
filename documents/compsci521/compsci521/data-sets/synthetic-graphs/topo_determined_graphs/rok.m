function [ A, str ] = rok( nK, mC ) 
  %
  %  SYNTAX
  %    A = rok( nK, mC, optional ); 
  %   
  % enerates the graph of Ring-of-Cliques (RoK) 
  %          all cliques are of equal size 
  % 
  % INPUT
  %   nK            integer > 0
  %                 the Clique size 
  %   mC            integer > 0
  %                 the cycle size
  % 
  % OUTPUT
  %
  %   A               adjacency matrix of the             [N x N]
  %                   'cliques on ring' graph             [sparse matrix]
  %                                                       [N = mC * nK]
    
    
  % ... generate block diagonals (Im \kron Kn)
  Kn = sparse(ones( nK ) - eye( nK ));
  
  
  Im = speye( mC );
  A  = kron( Im, Kn );
  
  if mC > 1
    % ... generate sparse up-shift/down-shift matrix
  
    colIdx = [( 2 : mC ).'; 1];
    rowIdx = ( 1 : mC ).';
    J = sparse(rowIdx, colIdx, ones( mC, 1), mC, mC );
  
    % ... generate single connection edge between cliques as a matrix
    C = sparse( floor( (nK+1)/3 ), floor( (nK+1)*2/3 ), 1, nK, nK );
  
    % ... the adjacency is A + kron(J, C) + kron(J.', C)
    A = A + kron(J, C) + kron(J.', C.');
  end
  
  n = size( A, 1 );
  X = [];
  
  str = sprintf('RoK(%d,%d)', nK, mC);
  
  
  return 
  
  %%------------------------------------------------------------
  %
  % AUTHORS
  %
  %   Tiancheng Liu               tcliu@cs.duke.edu
  %   Nikos Pitsianis             nikos@cs.duke.edu
  %   document revision by Xiaobai Sn
  % 
  % VERSION
  %
  %   0.1 - Oct 26, 2019
  %
  % CHANGELOG
  %
  %   0.1 (Oct 26, 2019) - Tiancheng
  %       * initial implementation
  %
  % ------------------------------------------------------------
  