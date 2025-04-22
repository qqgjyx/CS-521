% function demo_dim_reduction_tsne 

close all
clear all

% load fisheriris 
get_iris_data;               % with Lcolors for display 

X = meas; 

%% -- t-SNE: dimention reduction and spatial mapping  

method_name = 't-SNE: ' ; 
rng default           %  random-number generator, for reproducibility

Y2 = tsne(X);         % X --> Y2 

figure Name '2D-t-SNE'
scatter( Y2(:,1), Y2(:,2), 15, Lcolors, 'filled');
title( [ method_name, point_data_name] );


Y3 = tsne( X, 'NumDimensions', 3);   % X --> Y3 

figure 
scatter3( Y3(:,1), Y3(:,2), Y3(:,3), 15, Lcolors, 'filled')
title( [ method_name, point_data_name] );

[ ~, p1 ] = sort( Y3(:,1), 'ascend');
figure 
imagesc( X(p1, :)) 

return


%% ... with different metrics 

figure 
Y = tsne(meas,'Algorithm','exact','Distance','mahalanobis');
subplot(2,2,1)
gscatter(Y(:,1),Y(:,2),species)
title('Mahalanobis')

rng('default') % for fair comparison
Y = tsne(meas,'Algorithm','exact','Distance','cosine');
subplot(2,2,2)
gscatter(Y(:,1),Y(:,2),species)
title('Cosine')

rng('default') % for fair comparison
Y = tsne(meas,'Algorithm','exact','Distance','chebychev');
subplot(2,2,3)
gscatter(Y(:,1),Y(:,2),species)
title('Chebychev')

rng('default') % for fair
rng('default') % for fair comparison
Y = tsne(meas,'Algorithm','exact','Distance','euclidean');
subplot(2,2,4)
gscatter(Y(:,1),Y(:,2),species)
title('Euclidean')


fprintf('\n\n   demo finished \n\n')

return 

%% Programmer 
%% Xiaobai Sun 
%% Duke CS
