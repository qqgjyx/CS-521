%% 
% SCRIPT demo_Mycielski_embedding.m 
%
% -- generate a Mycielski matrix 
% -- display spectral embedding of the graph 
% -- plot the Laplacian eigenvalues 
% 
% 
% callee functions 
% 
% SGembedding3D 
% SGembedding_sparse 
% mycielski(...)   
% 

%% 

clear all 
close all 

addpath ../
addpath ../101-Graph-Generators 

fprintf( '\n\n   Begin of %s \n', mfilename ) ; 

k = input('\n   enter the recursion depth = ');

%      -------------- no need to change below --------------------

fprintf( '\n   generate the Mycielski graph of order %d \n', k );  

%%  ... construct the adajency matrix

A = mycielski( k ); 
n = size(A,1);
d = full( sum( A, 1 ) ) ;

figure 
semilogy( 1:n, d, 'b.' ) 

figure 
spy(A) 
axis equal off 
title( 'The adjacency matrix' ) 


%%  ... Laplacian spectral embedding 

ijk = [2,3,4] ; 
[Lambda, xyz] = SG_3D_embedding_basic( A, ijk  ) ;



fprintf( '\n\n   End of %s \n\n', mfilename ) ; 


% --------------------------------------------------------
% Xiaobai Sun, xiaobai@cs.duke.edu 
% 
% --------------------------------------------------------
