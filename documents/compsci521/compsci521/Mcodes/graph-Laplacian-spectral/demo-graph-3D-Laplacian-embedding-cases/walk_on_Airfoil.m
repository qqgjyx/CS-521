% walk_on_Airfoil.m 
% 
% This script does the following : 
% 1. loads the data of the airfoil mesh, provided by MATLAB, 
% 2. displays the mesh with the originally provided spatial node locations,
% 3. demonstrates a 'walk' or propagation on the graph, 
% 
% Callee functions (not built-in):
%   stepsGraph( ) 
% 
% Data 
%   airfoil          % matlab built-in collection 
% 

%% 

clear all; 
close all; 

fprintf('\n\n ... Begin of %s \n', mfilename  );

figID = 0; 

% ... load AIRFOIL with (i,j) edges and (x,y) node locations

load airfoil             % vectors i, j, x, y 
n = length(x);           % the number of nodes 

disp( sprintf('    AIRFOIL data loaded with %d nodes',n) );

% ... form the adjacency matrix, unweighted 

A = sparse(i,j,1,n,n); % a triangular portion of A 
A = A + A' ;           % symmetric 

% ... draw the mesh by provided spatial locations [x,y] 

figID = figID + 1; 
figure( figID) 
gplot(A, [x,y] ) ;
axis off equal 
title( 'Airfoid in the Original Triangulation Mesh(Submesh) ' )
hold on 

%%  ... walk or progation on the graph 

fprintf('\n ... pick randomly a few departure nodes to walk from \n')

uindex = randperm(n); 
m = 10 ;                   % number of departunre nodes 
m = input('    enter the number of departure nodes = '); 

uindex = uindex(1:m); 
u = zeros(n,1);
u( uindex) = 1; 

k = 5;                     % number of steps to take 
k = input('    enter the number of walk steps  = '); 
v = stepsGraph( A, k, u, 'walk' ); 
vindex = find(v); 
mv = length( vindex );

figID = figID + 1;
figure( figID ) 

subplot(2,1,1) 

gplot(A, [x,y], 'c');     % the original graph 
axis off 
hold on 
% ... highlight the source/departure nodes 
gplot( ones(m,m), [x(uindex), y(uindex)], 'bo' );
title( 'Departure Nodes' ) 

subplot(2,1,2) 

gplot(A, [x,y], 'c' ); % the graph 
axis off 
hold on 
% ... highlight arrival nodes 
gplot( ones(mv,mv) , [x(vindex), y(vindex)], 'mo' ); 
title( sprintf('Arrival Nodes by %d Steps Walking', k) ) ;  


%% 
fprintf('\n\n ... End of %s \n\n', mfilename  );

return 

% End of the file 

%%  --------------------------------------------
% Xiaobai Sun 
% Duke CS 
% For further distribution, constact first 
%       xiaobai@cs.duke.edu 
% Initial draft: Oct.  2011 
% Last revision: Sept. 2021 
% ---------------------------------------------