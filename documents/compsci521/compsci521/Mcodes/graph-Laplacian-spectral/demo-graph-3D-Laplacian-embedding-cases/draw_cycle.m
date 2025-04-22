% draw_cycle.m
% 
% to draw a cycle via spectral embedding 
% 
% Callee functions (not matlab built-in) : 
% gen_cycle(...)  
% gplot3D(...) 
% 
% Examples : n=5, n=86 

clear all ; 
close all ; 

fprintf('\n\n ... Begin of %s \n', mfilename  ); 

n = input( '   Enter the cycle length n > 2 :  ' );

% suggestions : try d = 2, 3, 4, 8, 10 

figID = 0; 

Adisplay  = input( '   Select 1 to display the cycle generation =  ' );

%%  -------- main driver ---------------------------

fprintf('\n\n ... generate the adjancency matrix \n'); 

% ... generate the adjacency matrix 

if Adisplay == 0 
  A = gen_cycle( n );
else 
  figID = figID + 1; 
  A = gen_cycle( n, figID );
end
clear Adisplay 

%% ... form the Laplacian matrix 

fprintf('\n\n ... press a key for spectral embedding \n'); 
pause( )

fprintf('\n\n ... get the Laplacian spectral decomposition \n'); 

D = sum( A, 2 ) ;           % row sums = node dgrees 
L = diag( D ) - A ;       

%%  ... get the Laplacian spectral decomposition 

[ U, Lambda ] = eig( full(L) ); 
Lambda = diag( Lambda ) ;   % change the data structure 
Lambda = real( Lambda ) ;

[ Lambda, Jpermute ]  = sort( Lambda, 'ascend' );
U = U(:, Jpermute ) ;       % re-order the eigenvectors accordingly 

%% ------------ Specral embedding of the graph --------------- 

fprintf('\n\n ... display the graph in 2D spectral embedding \n'); 

ij = 1+[1,2]; 
Sxy = U(:, ij); 
figure 
gplot( A, Sxy,  'b' );
axis image off 
title( sprintf('Cycle C(%d) in 2D spectral embedding at the low end', n) );

figure 

nij = [n-1,n];
Sxy = U(:, nij); 
gplot( A, Sxy,  'm' );
axis image off 
title( sprintf('Cycle C(%d), a bipartite, in 2D spectral embedding at the high end', n) );

if mod(n,2) == 0 
    nij = [n-2,n-1];
    Sxy = U(:, nij);
    figure 
    gplot( A, Sxy,  'c' );
    axis image off 
    title( sprintf('Cycle C(%d) in 2D spectral embedding near the high end', n) );
end

fprintf('\n\n ... plot the Laplacian eigenvalues \n');

figure 
plot( Lambda, 'm*' ) 
BannerStr = sprintf( 'The Laplacian eigenvalues of graph C(%d)', n ); 
xlabel( ' spectral index' ) ; 
ylabel( ' spectral values' );
title( BannerStr ); 

fprintf('\n\n *** If you see a pattern in the eigenvalue disribution, try to prove it :) \n'); 

%%
fprintf('\n\n ... End of  %s \n\n', mfilename  ); 

return 

% ---------------------------------------------------
% Xiaobai Sun 
% Duke CS 
% For Numerical Analysis Class 
% Last revision: Sept. 2021 
% Contact xiaobai for any further distribution 
% ---------------------------------------------------
