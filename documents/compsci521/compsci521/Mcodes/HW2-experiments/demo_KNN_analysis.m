% function demo_KNN_analysis 
% 
%  demonstrate analysis of feature vector data via kNN graph 
%  and Laplacian spectral information for 
%    -- feature vectors -->  graph vertices 
%    -- graph vertices  -->   graph vertex encoding/embedding  
%                       -->  vertex clustering 

close all
clear all

%% .. get the data 

fprintf('\n\n   %s began', mfilename);

fprintf('\n   load a feature-vector dataset ...')
get_iris_data;                         % with colorlables  
X  = meas ;                            % X is the orignal feature data array 
n  = size( X, 1);

%% ... set the analysis options 

fprintf('\n   setup KNN analysis options: ');

k           = input( '\n   specify #neighbors k = ') ; 
[name_metric, p_mink_val] = metric_selection;
%% ... make the knn analysis via knn graph and its Laplacian-alpha 
fprintf('\n   construct kNN graph ... ');

A_knn = knn_graph_construction ( X, k, name_metric, p_mink_val  ) ;

fprintf('\n   make Laplacian-spectral analysis ...');

alpha = 0.95;
dimY  = 3;

[ piFiedler, idxcut, Y ]  = Laplacian_alpha_Vembedding( A_knn, alpha, Lcolors, dimY ) ;

Xp = X( piFiedler,:);
Xp( idxcut, :) = 1;

%% 
if n < 2500 
    fprintf('\n   display knn matrix in Fiedler ordering ...');
    figure 
    imagesc(A_knn( piFiedler, piFiedler), 'AlphaData', A_knn( piFiedler, piFiedler) > 0);
    axis equal tight 
    colorbar
    colormap( gca, flipud(parula(5)) )
    title('A\_knn in Fiedler order')
end

%%
fprintf('\n   display feature data in Fiedler ordering ...');
figure
subplot(1,2,1)
imagesc( X )
title('Feature vector data')
ylabel('initial order')
% 
subplot(1,2,2)
imagesc( Xp )
ylabel('Fiedler order')

%%
Brecurse = input('\n   make a recursive cut? [1 or 0] = ');

if ~Brecurse 
    fprintf('\n\n   %s finished \n\n\n', mfilename);
    return 
end

%% ... take a cut/divided subset of the feature vector data 

X1 = Xp(1:idxcut, : );
X2 = Xp(idxcut+1:n, : ); 

Lcolorsp = Lcolors( piFiedler, 1:3) ;
Lcolors1 = Lcolorsp( 1:idxcut, 1:3); 
Lcolors2 = Lcolorsp( idxcut+1:n, 1:3);

jsub = input('\n   choose a subset for recursion (1 or 2) = ');
if jsub == 1 
  A1_knn = knn_graph_construction ( X1, k, name_metric, p_mink_val  ) ;
  [ piFiedler1, idxcut1 ]  = Laplacian_alpha_Vembedding( A1_knn, alpha, Lcolors1, dimY ) ;

  X1 = X1( piFiedler1, : );
  X1( idxcut1, :) = 1;
elseif jsub == 2  
  A2_knn = knn_graph_construction ( X2, k, name_metric, p_mink_val  ) ;
  [ piFiedler2, idxcut2 ]  = Laplacian_alpha_Vembedding( A2_knn, alpha, Lcolors2, dimY ) ;

  X2 = X2( piFiedler2, : );
  X2( idxcut2, :) = 1;
end

%%
figure 
subplot(1,3,1)
imagesc( X ) 
title('Feature vector data')
ylabel('initial order')
% 
subplot(1,3,2)
imagesc( Xp ) 
ylabel('Fiedler order') 
% 
subplot(1,3,3)
imagesc( [X1; X2] ) 
ylabel('recurive Fiedler order')

%% 
fprintf('\n\n   %s finished \n\n\n', mfilename);
 
return 

%% programmer 
%% Xiaobai Sun 
%% Last revision: Oct. 22, 2024 
%% 