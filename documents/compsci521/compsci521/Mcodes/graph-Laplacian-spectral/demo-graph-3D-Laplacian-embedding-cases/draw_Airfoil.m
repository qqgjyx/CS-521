% draw_Airfoil.m 
% 
% This script does the following : 
% 1. loads the data of the airfoil mesh, provided by MATLAB, 
% 2. displays the mesh with the originally provided spatial node locations,
% 3. demonstrates a 'walk' or propagation on the graph, 
% 3. displays the mesh with 2D special embedding of the graph, 
%    or its subgraphs, in different 2D spectral spaces 
% 
% Callee functions (not built-in):
%   stepsGraph( ) 
%   SGembedding( ) 
% Data 
%   airfoil          % matlab built-in collection 


%% 

clear all; 
close all; 

fprintf('\n\n ... Begin of %s \n', mfilename  );

figID = 0; 

%% ... load AIRFOIL with (i,j) edges and (x,y) node locations

load airfoil             % vectors i, j, x, y 
n = length(x);           % the number of nodes 

disp( sprintf('     AIRFOIL data loaded, #nodes = %d',n) );

% ... form the adjacency matrix, unweighted 

A = sparse(i,j,1,n,n); % a triangular portion of A 
A = A + A' ;           % symmetric 

%% ... draw the mesh by provided spatial locations [x,y] 

figID = figID + 1; 
figure( figID) 
gplot(A, [x,y] ) ;
axis off equal 
title( 'Airfoid in the Original Triangulation Mesh(Submesh) ' )


%%  ... display the mesh with Laplacian spectra embedding   

dim = '3D'; 

figID = figID + 1; 
Sxy = SGembedding_sparse(A, dim, figID, 0 );   
figure( figID ) 
view( [-128, 16]  ) 

%%  How to get, record and repeat the current rotation view:
% select the figure by its ID 
% [AZ, EL] = view; 
% record/reset the view by view( [AZ, EL] ) 

fprintf('\n * * * observe and comment on the two embeddings ' );

%% 
fprintf('\n\n ... End of %s \n\n', mfilename  );

% End of the file 

%%  --------------------------------------------
% Xiaobai Sun 
% Duke CS 
% For further distribution, constact first 
%       xiaobai@cs.duke.edu 
% Initial draft: Oct.  2011 
% Last revision: Sept. 2021 
% ---------------------------------------------