% draw_Bucky_sparse.m 
% Objective 
%  -- demonstrate spectral embedding of an undirected graph 
%  -- show the spectral coordinates for each vertex 
%  -- highlight the frequency impact on pairwise relationship 
%     among vertices in a spatial embedding 
% 
% Callee function 
%     SGembedding_sparse(...) which renders two images
% 
% Built-in data 
%     Bucky 
% ------------------------------------------------------------------

%% 
close all 
clear all 

fprintf('\n\n ... Begin of %s \n', mfilename  ); 

%% ... load the data provided by MATLAB 

[ A, Cxyz] = bucky;                              
                        
n = size(A,1);          
fprintf( '\n  n = %d nodes ', n );

%% 


fprintf('\n\n ... 2D spectral embedding \n' ); 
figID = 1 ; 
Sxyz  = SGembedding_sparse( A, '2D', figID);

fprintf('\n\n ... press a key to proceed to 3D spectral embedding \n' );
pause()


fprintf('\n\n ... 3D spectral embedding \n' ); 
figID = figID + 2 ; 
Sxy  = SGembedding_sparse( A, '3D', figID);



%%
fprintf('\n\n ... End of %s \n\n', mfilename  ); 

return


% END of the script file 

%% ======= Programmer info 
% Xiaobai Sun 
% Duke CS 
% Last revision: Sept. 2021 
% For the use of Numerical Analysis Class 