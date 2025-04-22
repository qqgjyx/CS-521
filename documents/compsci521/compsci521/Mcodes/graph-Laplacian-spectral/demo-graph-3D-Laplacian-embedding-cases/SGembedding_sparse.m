function [ Sdim] = SGembedding_sparse( A, dim,  figID, colorNodes) 
% 
% calling sequence 
%      Sxyz = SGembedding_sparse(A, dim, figID, colorNodes ) ; 
% 
% A  : symmetric, non-negative elementwise, irreducible, 
%      with zero diagonal elements 
%      representing an undirected, connected graph without self-loops 
% 
% dim:    character string in {'2D',  '3D'} 
% nodeColor:  Boolean { 0, 1 } 
% 
% The function displays in Figure( figID ) and Figure ( figID + 1) 
% the dim-spatial embedding of graph G(A) at each of the spectral ends;  
% it also returns the eigenvectors 
% 
% Sdim returns the spectral values and vectors in 
%           Sdim.lowvals 
%               .lowvecs 
%               .highvals 
%               .highvecs 
%      
% When the graph is numerically disconnected, this function 
% returns with a message on numerical disconnection. 
% 
% Method: 
%  use sparse eigen-solver EIGS(...) for extrme Laplacian eigenvalues 
%  and eigenvectors; a sparse counterpart of SGembedding3D 
% 
% Examples:  
%    Sxy   = SGembedding_sparse( A, '2D', figID);
%    Sxyz  = SGembedding_sparse( A, '3D', figID); 
% 


% --------------------------------------------

fprintf( '\n ... in %s \n' , mfilename );

if min( A(:) ) < 0 
   error( 'there are negative elements' );
end

if norm( diag(A), 'inf' ) > 0 
    error( 'there are nonzero diagonal elements/self-loops' ) ; 
end

if nargin < 4 
  showNodes = 1;
else
  showNodes = colorNodes; 
end

% ... find the degree and form the Laplacian matrix 

n = size(A,1);        % the number of nodes 
m = nnz(A)/2;         % the number of edges 

D = sum(A,1);         % D(i) = the degree of node i 
L = diag(D) - A ;     % form the Laplancain matrix 

% ... get the extreme eigenvalues and eigenvectors of the Laplacian 

Sigma = 'bothendsreal'; 

k2 = 20;      % Hard coded to avoid numerical failure in convergence 
              % with EIGS(...) which often render unstable eigenvectors 
              % when k2 is small 

[ V, S ] = eigs(L, k2, Sigma);  

% ... sort the eigenvalues 

S     = diag(S);            % place the eigenvalues into a vector  
S     = real(S);            % recast by theory; a longstanding problem MATLAB
V     = real(V); 


[S,P] = sort(S, 'ascend' ); % sort the eigenvalues non-decreasingly
V     = V(:,P);             % permuate the eigenvectors accordingly 

if ( abs( S(2)/S(k2) ) < 100*eps ) 
    disp( '    ==> the graph is numerically disconnected'); 
    return 
end

switch dim 
  case '3D'

    % ... display the 3D embedding at the low spectral end 
    
    ijk  = [2,3,4];
    Sxyz = V(:,ijk); 

    Sdim.lowvals = S(ijk); 
    Sdim.lowvecs = Sxyz; 
    
    
    figure( figID ) 
    clf 
    if showNodes 
     gplot3D( A, Sxyz , 'mx' );         % display the nodes 
     hold on 
    end 
    gplot3D( A, Sxyz, 'b-' );          % display the edges 
    axis equal off  

    tBanner = [ 'Graph embedding in ' ]; 
    tBanner = [ tBanner, sprintf(' the (%d,%d,%d) spectral space', ijk)] ;
    disp( ['     ', tBanner] ) 

    tBanner = [tBanner, sprintf(' with %d nodes and %d edges', n,m) ]; 
    title( tBanner ) ;
    
    hold off
    rotate3d

    % ... display the 3D embedding at the high spectral end 
    
    ijk  = [ k2-2, k2-1, k2];
    Sxyz = V(:,ijk); 
    
    Sdim.highvals = S(ijk); 
    Sdim.highvecs = Sxyz; 
        
    figure( figID+1 ) 
    clf
    if showNodes  
      gplot3D( A, Sxyz , 'mx' );       % display the nodes 
      hold on 
    end 
    gplot3D( A, Sxyz, 'b-' );          % display the edges 
    axis equal off  

    tBanner = [ 'Graph embedding in ' ]; 
    tBanner = [ tBanner, sprintf(' the (%d,%d,%d) spectral space', (n-2:n)) ] ;
    disp( ['     ', tBanner] ) 

    tBanner = [tBanner, sprintf(' with %d nodes and %d edges', n,m) ]; 
    title( tBanner ) ;
    
    hold off
    rotate3d

  case '2D'

    % ... display the 2D embedding at the low spectral end 
    
    ij   = [2,3];
    Sxy  = V(:,ij); 
    Sdim.lowvals = S(ij); 
    Sdim.lowvecs = Sxy; 
    
    figure( figID ) 
    clf 
    if showNodes 
      gplot3D( A, Sxy , 'mx' );         % display the nodes 
      hold on 
    end 
    gplot3D( A, Sxy, 'b-' );          % display the edges 
    axis equal off  
    
    tBanner = [ 'Graph embedding in ' ]; 
    tBanner = [ tBanner, sprintf(' the (%d,%d) spectral space', ij)] ;
    disp( ['     ', tBanner] ) 

    tBanner = [tBanner, sprintf(' with %d nodes and %d edges', n,m) ]; 
    title( tBanner ) ;
    
    hold off
    rotate3d

    % ... display the 2D embedding at the high spectral end 
    
    ij  = [ k2-1, k2];
    Sxy = V(:,ij); 
    Sdim.highvals = S(ij); 
    Sdim.highvecs = Sxy; 
    
    
    figure( figID + 1 ) 
    clf 
    if showNodes 
      gplot( A, Sxy , 'mx' );         % display the nodes 
      hold on 
    end 
    gplot( A, Sxy, 'b-' );            % display the edges 
    axis equal off  

    tBanner = [ 'Graph embedding in ' ]; 
    tBanner = [ tBanner, sprintf(' the (%d,%d) spectral space', [n-1,n]) ] ;
    disp( ['     ', tBanner] ) 

    tBanner = [tBanner, sprintf(' with %d nodes and %d edges', n,m) ]; 
    title( tBanner ) ;
    
    hold off
    rotate3d

end


return

% ---------------------------------------
% Xiaobai Sun, Duke CS
% Initial draft: 
% Last draft: Sept. 2021 
% ----------------------------------------

