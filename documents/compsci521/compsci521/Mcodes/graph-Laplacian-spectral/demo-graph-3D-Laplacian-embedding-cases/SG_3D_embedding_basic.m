function [ Lambda, xyz]  = SG_3D_embedding_basic( A, ijk ) 
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

% --------------------------------------------

fprintf( '\n     ... %s begin  \n', mfilename  ); 

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

S     = real( diag(S) ) ;            % place the eigenvalues into a vector  
V     = real(V);

[S,P] = sort(S, 'ascend' ); % sort the eigenvalues non-decreasingly
V     = V(:,P);             % permuate the eigenvectors accordingly 

figure 
semilogy( 1:n, 1+S, 'm.' ) ; 
title( 'Laplacian eigenvalues');


if ( abs( S(2)/S(n) ) < 100*eps ) 
    disp( '    ==> the graph is numerically disconnected'); 
    return 
end


%%  output variables 

Lambda = S; 
xyz    = V(:,ijk); 

%% ... display the embedding 

figure 
 gplot3D( A, xyz , 'mx' );        % display the nodes 
 hold on 
 gplot3D( A, xyz, 'b-' );          % display the edges 

axis equal off 
tBanner = [ 'Graph embedding in ' ]; 
tBanner = [ tBanner, sprintf(' the (%d,%d,%d)-spectral space', ijk)] ;
title( tBanner ) ;
    


return

% ---------------------------------------
% Xiaobai Sun, Duke CS 
% ----------------------------------------

