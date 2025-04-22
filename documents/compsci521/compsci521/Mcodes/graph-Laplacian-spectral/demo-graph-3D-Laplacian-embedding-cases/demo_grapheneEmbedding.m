%% 
% SCRIPT demo_grapheneEmbedding.m 
%
% -- generate a graphne matrix 
% -- display spectral embedding of the graph 
% -- plot the Laplacian eigenvalues 
% 
% 
% callee functions 
% 
% SGembedding3D 
% SGembedding_sparse 
% gen_grapheneSheet  
% 

%% 

clear all 
close all 

fprintf( '\n\n   Begin of %s \n', mfilename ) ; 

demo_case = input( '    select a demo case [1,2, 0] = ') ; 
disp( ' ') 

switch demo_case  
 case 1 
   ny = 31; 
   nx = 51; 
 case 2 
  ny = 65; 
  nx = 70;
   
  otherwise  
    ny = input( '    enter #lattice-rows    = ' ); 
    nx = input( '    enter #lattice-columns = ' ); 
end

n = nx*ny; 

%      -------------- no need to change below --------------------

Dmsg = sprintf(' %d Y-shape nodes in %d rows and %d columns ', ...
                 n, ny, nx ); 

disp( [ '    generating a graphene sheet of', Dmsg ] ) 


%%  ... construct the adajency matrix

A = gen_grapheneSheet( ny, nx ) ; 

figure 
spy(A) 
axis equal off 
title( 'The adjacency matrix' ) 


%%  ... Laplacian spectral embedding 

figID = 1; 

if n < 1024*3   % hard-coded upper limit 
   
   Sxyz = SGembedding3D(A, [2,3,4], figID, 0 ) ;  % at low spectral end
   
   Sxyz = SGembedding3D(A, [n-2,n-1,n], figID+2, 0 ) ; % at high spectral end 
   
else 
   Sxyz = SGembedding_sparse( A, '3D',  figID, 0 ) ; 
end 


fprintf( '\n\n   End of %s \n\n', mfilename ) ; 


% --------------------------------------------------------
% Nikos P. Pitsianis, nikos@cs.duke.edu, 
% Xiaobai Sun, xiaobai@cs.duke.edu 
% 
% Initial version: January 10, 2012 
% Lat revision:    Spet. 2021 
% 
% Contact xiaobai for permission of using the functions 
% and material for purpose other than the homework 
% --------------------------------------------------------
