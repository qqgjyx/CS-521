% draw_Bucky3D.m 
% 
% Objectives 
%  -- demonstrate spectral embedding of an undirected graph 
%  -- show the spectral coordinates for each vertex 
%  -- highlight the frequency impact on pairwise relationship 
%     among vertices in a spatial embedding 
% 
% Callee function 
%     SGembedding3D(...) which renders two images
% 
% Built-in data 
%     Bucky 
% ------------------------------------------------------------------

%% 
close all 
clear all 

fprintf('\n\n ... Begin of %s \n', mfilename  ); 

%% ... load the data provided by MATLAB 

[ G, Cxyz] = bucky;                              
                        
n = size(G,1);          
fprintf( '\n  n = %d nodes ', n );

%% 

fprintf('\n\n ... spatial embedding om low-frequency eigenspace \n' ); 
figID = 1; 
ijk   = [2:4];          % low spectral indices chosen for spectral embedding 
Sxyz  = SGembedding3D( G, ijk, figID);

fprintf('\n\n ... spatial embedding om low-frequency eigenspace \n' ); 

figID = figID + 2; 
ijk   = [n-2:n];          % high spectral indices chosen for specral embedding 
Sxyz  = SGembedding3D( G, ijk, figID); 

fprintf('\n\n ... rotate each plot for inspection from different view angles \n' );

%% ... try some other eigen space (i,j,k) 

FlagNext = input( '\n  Continue with another embedding ? [y/n] ', 's');  

while FlagNext == 'y' 
    ijk(1) = input( ' Enter the index for the first  axis ' );
    ijk(2) = input( ' Enter the index for the second axis ' );
    ijk(3) = input( ' Enter the index for the third  axis ' );
    % try with [2,5,10]
    
    figID = figID + 1; 
    Sxyz = SGembedding3D( G, ijk, figID );
    FlagNext = input( '\n  Continue ? [y/n] ', 's'); 
    
end

%%
fprintf('\n\n ... End of %s \n\n', mfilename  ); 

return 


% END of the script file 

%% ======= Programmer info 
% Xiaobai Sun 
% Duke CS 
% Last revision: Sept. 2021 
% For the use of Numerical Analysis Class 