% SCRIPT: demo_coordinate_transform.m 
% 
% This script does the following : 
% 1. load the data of the airfoil mesh, provided by MATLAB, 
% 2. display the mesh with the originally provided spatial node locations,
% 3. make a randomized rigid change of the coordinate grid 
% 
% Callee function
% draw_grid_transformation(...)
% 
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

%% ... make a randomized rigid transform 

fprintf('\n ... make and show a randomized rigig coordinate transform \n' );

T = rand(2,2); 
draw_grid_transformation( A, T, [x,y] );

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