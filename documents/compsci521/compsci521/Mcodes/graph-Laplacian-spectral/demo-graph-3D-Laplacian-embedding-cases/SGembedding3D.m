function [ Sxyz] = SGembedding3D( A, ijk, figID, colorNodes ) 
% 
% calling sequence 
%      Sxyz = SGembedding3D(A, ijk, figID, nodeColor ) ; 
% 
% The function renders three displays in three figures 
% starting with Figure( figID ) 
% -- the 3D spatial embedding in the spectral space specified by 
%    tbe spectral index triple ijk. 
%    The eigenvalues are indexed in non-decreasing order.
% 
% -- the spectral coordinates of the graph vertices are also displayed 
%    if the graph size is modest; the hard-coded upper limit is 128 nodes 
% 
% -- The Laplacian eigenvalues 
% 
% If nodeColor == 1 the nodes are shown with a different color. 
% 
% When the graph is numerically disconnected, this function 
% gives a message. 
% 
% Examples:  
% 
%     SGembedding3D( A, [2:4], 1) ;
% 
%     SGembedding( A, [n-2:n], 1 ); % n = size(A,1) 
% 


% --------------------------------------------

fprintf( '\n     ... in SGembedding3D \n' );

if nargin < 4 
    showNodes = 1;
else
    showNodes = colorNodes ;
end

% ... find the degree and form the Laplacian matrix 

n = size(A,1);        % the number of nodes 
m = nnz(A)/2;         % the number of edges 

D = sum(A,1);         % D(i) = the degree of node i 
L = diag(D) - A ;     % form the Laplancain matrix 
L = full(L);          % sparse version eigs has 
                      % unfortunate limitation for this purpose

% ... find the eigenvalues and eigenvectors of the Laplacian 

[ V, S] = eig(L);  

% ... sort and check on the second eigenvalue 

S     = diag(S);            % place the eigenvalues into a vector  
S     = real(S);            % recast to real: long-problem with MATLAB eig
V     = real(V);

[S,P] = sort(S, 'ascend' ); % sort the eigenvalues non-decreasingly
V     = V(:,P);             % permuate the eigenvectors accordingly 

if ( abs( S(2)/S(n) ) < 100*eps ) 
    disp( '    ==> the graph is numerically disconnected'); 
    return 
end

Sxyz = V(:,ijk); 

% ... display the embedding 

figure( figID ) 
clf 

if showNodes 
 gplot3D( A, Sxyz , 'mx' );        % display the nodes 
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

if n < 129   % HARD-coded upper limit  
 A2 = [ zeros(n), eye(n); eye(n), zeros(n) ]; 
 Sx = [ Sxyz ; [ Sxyz(:,1), -ones(n,1)/2, -ones(n,1)/2 ] ]; 
 Sy = [ Sxyz ; [ -ones(n,1)/2, Sxyz(:,2), -ones(n,1)/2 ] ]; 
 Sz = [ Sxyz ; [ -ones(n,1)/2, -ones(n,1)/2, Sxyz(:,3) ] ]; 

 figure 
 gplot3D( A, Sxyz, 'b-' );
 grid on 
 axis equal

 hold on 
 view( [ -36.5, -22] )
 gplot3D( A2, Sx, 'm-.' ); 
 pause(2) 
 gplot3D( A2, Sy, 'g-.' );
 pause(2) 
 gplot3D( A2, Sz, 'c-.' ); 
 pause(2) 

 rotate3d 

 tBanner = [ 'Graph embedding in ' ]; 
 tBanner = [ tBanner, sprintf(' the (%d,%d,%d) spectral space', ijk)] ;
 tBanner = [ tBanner, sprintf(' with spectral coordinates shown') ]; 
 title( tBanner ) ;
end

figure 
plot( S, 'm*' ) 
title( 'Laplacian eigenvalues');

return

% ---------------------------------------
% Xiaobai Sun, Duke CS 
% For the class of Numerical Analysis 
% ----------------------------------------

