% draw_Dcube.m
% 
% to draw a hypercue and demonstrate the relationship 
% between d-dim cubs and its two subcubes of (d-1) dimension
% 
% Callee functions (not matlab built-in) : 
% gen_Dcube(...)  
% gplot3D(...) 


clear all ; 
close all ; 

fprintf('\n\n ... Begin of %s \n', mfilename  ); 

d = input( '   Enter the hypercube dimension d => 2 :  ' );

% suggestions : try d = 2, 3, 4, 8, 10 

figID = 0; 

Adisplay  = input( '   Select 1 to display the Dcube generation =  ' );

%%  -------- main driver ---------------------------

fprintf('\n\n ... generate the adjancency matrix \n'); 

% ... generate the adjacency matrix 

if Adisplay == 0 
  A = gen_Dcube( d );
else 
  figID = figID + 1; 
  A = gen_Dcube( d, figID );
end
clear Adisplay 

%% ... form the Laplacian matrix 

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



%% ... get two sub-cubes for displaying purpose 
    
    n2 = 2^(d-1);
    A1 = A(1:n2, 1:n2 );               
    
    B = A; 
    B( n2+(1:n2), n2+(1:n2) ) = A( n2+(1:n2), n2+(1:n2) ) - A1; 
    C = A - B; 
    
    B( 1:n2, 1:n2 ) =  A( 1:n2, 1:n2 ) - A1;   
       
    %%  ... choose low-energe eigenvectors as the embedding axses 
    
    xy = U(:, 1+[1, 2] );
    
    %% ... 2D display 
    
    figID = figID + 1; 
    figure( figID ) 

    gplot( A1, xy, 'g');     % subcube-1 
    hold on
    gplot( C, xy, 'm' );     % subcube-2 
    gplot( B, xy, 'b' );     % inter-connections 
    
    axis off image 
    title( sprintf( ' 2D Embedding of the %d-dim Hypercube and two subcubues \n ', d )) ;
    
    
    
   if d > 2 
       
    fprintf('\n\n ... display the graph in 3D spectral embedding \n'); 
    
    %%  ... choose low-energe eigenvectors as the embedding axses 
    xyz = U(:, 1+[1, 2, 3] ); 
    
    figID = figID + 1; 
    figure( figID ) 
    
    gplot3D( A1, xyz, 'g' );  % subcube-1 
    hold on 
    gplot3D( C, xyz, 'm' ) ;  % subcube-2 
    axis off image 
    gplot3D( B, xyz, 'b' ) ;  % inter-connection 

    axis off image 
    title( sprintf( ' 3D Embedding of the %d-dim Hypercube and two subcubes \n ', d )) ; 
       
    rotate3d on 
    disp( sprintf( ' \n --> rotate the graph in Figure-%d \n', figID ) ); 

   % for d=4, one shall see two perfect cubes at certain view 

end 

fprintf('\n\n ... plot the Laplacian eigenvalues \n');

figID = figID + 1; 
figure( figID ) 

plot( Lambda, 'm*' ) 
BannerStr = sprintf( 'The Laplacian eigenvalues of the %d-Hypercube', d ); 
xlabel( ' spectral index' ) ; 
ylabel( ' spectral values' );
title( BannerStr ); 

fprintf('\n\n *** If you see a pattern in eigenvalue multiplicities, try to prove it :) \n'); 

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
