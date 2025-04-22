function [ Sxyz] = SGembedding3D( A, ijk, figID) 
% calling sequence 
%      Sxyz = SGembedding3D(A, ijk, figID ) ; 
% 
% It displays in the figure specified by figID 
% the graph represented by the adjacent matrix A, 
% in the spectral subspace with the dimensions 
% specified in ijk > 1 as the embedding axies. 
% 
% The eigenvalues are in non-decreasing order.
% 
% It also detects and tells whether or not 
% tne graph is disconnected. 
% 
% Examples for empirical observation : 
% 
%     SGembedding3D( A, [2:4], 1) ;
% 
%     SGembedding( A, [n-2:n], 1 ); % n = size(A,1) 
% 


% --------------------------------------------

fprintf( '\n  SGembedding3D ... ' );

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
[S,P] = sort(S, 'ascend' ); % sort the eigenvalues non-decreasingly
V     = V(:,P);             % permuate the eigenvectors accordingly 

if ( abs( S(2)/S(n) ) < 100*eps ) 
    disp( '  the graph is numerically disconnected'); 
    return 
end

Sxyz = V(:,ijk); 

% ... display the embedding 

figure( figID ) 
clf 
gplot3D( A, Sxyz , 'mx' );         % display the nodes 
axis equal off 
hold on 

gplot3D( A, Sxyz, 'b-' );          % display the edges 

tBanner = [ 'Graph embedding in ' ]; 
tBanner = [ tBanner, sprintf(' the (%d,%d,%d) spectral space', ijk)] ;
disp( tBanner ) 

tBanner = [tBanner, sprintf(' with %d nodes and %d edges', n,m) ]; 
title( tBanner ) ;
    
hold off
rotate3d

A2 = [ zeros(n), eye(n); eye(n), zeros(n) ]; 
Sx = [ Sxyz ; [ Sxyz(:,1), -ones(n,1)/2, -ones(n,1)/2 ] ]; 
Sy = [ Sxyz ; [ -ones(n,1)/2, Sxyz(:,2), -ones(n,1)/2 ] ]; 
Sz = [ Sxyz ; [ -ones(n,1)/2, -ones(n,1)/2, Sxyz(:,3) ] ]; 

%%  ... show coordinates 

figure 
gplot3D( A, Sxyz, 'b-' );
axis equal 
box on ; grid on 
xlabel('x')
ylabel('y') 
zlabel('z')


hold on   % to add coordinate projection lines 

view( [ -36.5, -22] )
gplot3D( A2, Sx, 'm-.' ); 
pause(2) 
gplot3D( A2, Sy, 'g-.' );
pause(2) 
gplot3D( A2, Sz, 'c-.' ); 
pause(2) 

 
rotate3d 

return

% ---------------------------------------
% Xiaobai Sun, Duke CS 
% For the class of Numerical Analysis 
% ----------------------------------------

