function A_knn = knn_graph_construction ( X, k, name_metric, p_mink_val  ) 
% 
%  A_knn = knn_graph_construction ( X, k, name_metric  ) ;
% 
%  INPUT 
%  X: n x d array, n feature vector of length d 
%  k: inteter, positive, specifyng the number of nearest neighbors  
%  name_metric: character string, the name of pairwise discrepancy measure 
% 
%  OUTPUT 
%  A_knn: n x n, non-negative, real-valued 
%         as the adjancy matrix for k-nearest-neighbor graph 
%         using a Gaussian kernel as the defaul to convert 
%         the pairwise disrepancy to pairwise dissimilarity 
% 
%  Note:
%        the kNN information is in two n x k data arrays 
% 


%% ... get the k nearest neighbors 

switch name_metric  
    case 'minkowski'    % norm(x-y, p) 
      [ Idx_knn, D_knn ]  =  ... 
          knnsearch( X, X, 'K', k, 'Distance', name_metric, 'P', p_mink_val );  
    otherwise 
      [ Idx_knn, D_knn ] =  knnsearch( X, X, 'K', k, 'Distance', name_metric );
end


%% ... converting pairwise discrepancy to dis-similarity : 
%      using a Gaussian kernel by default, the scale parameter sigma is data dependent

sigma  = 2*std( D_knn(:) );   % uniform scale, shall and can be modified to be adaptive 
                              % sigma(i) = max( D_knn(i,:) ) % SD-DP (2018)
                           
gD_knn = exp( -(D_knn./sigma).^2/2 );

n = size(X,1);

figure 
subplot(1,2,1) 
imagesc(D_knn);
xlabel('Discrepancy/distance') ; 
colorbar 
tmsg = sprintf('%d nearest neighbors', k);   
title( tmsg ) 
%
subplot(1,2,2) 
imagesc( gD_knn );
ylabel('data points')
xlabel('(dis)similarity') ;
colorbar 

%% ... format to adjacency matrix 

A_knn = knn2adjacency( Idx_knn, gD_knn  ) ; 

%%% ... display 

din =  A_knn  * ones(n,1);
dout = A_knn' * ones(n,1); 

[ din, pdin] = sort( din, 'ascend');
dout = dout( pdin );

figure 
plot( 1:n, din, 'm+', 1:n, dout, 'bo');
legend('in-degree','out-degrees');
tmsg = sprintf( 'Weighted degrees of KNN graph (k=%d)', k ); 
title( tmsg )

if size( Idx_knn, 1) < 2500   % change pixel scale otherwise 
     figure
     subplot(1,2,1)
     % imagesc( knn2adjacency( Idx_knn, D_knn  )  ) ;
     A_knn_0 = knn2adjacency( Idx_knn, D_knn  );
     imagesc( A_knn_0, 'AlphaData', A_knn_0 > 0 )
     axis equal tight
     colorbar
     colormap( gca, flipud(parula(5)) )
     
     tmsg = sprintf( 'KNN distance matrix (k=%d)', k);
     title( tmsg ); 
     subplot(1,2,2)
     imagesc( A_knn, 'AlphaData', A_knn > 0 )
     
     axis equal tight
     colorbar
     colormap( gca, flipud(parula(5)) )
     tmsg = sprintf( 'KNN (dis)similairty matrix (k=%d)', k);
     title( tmsg );
 end 
 
return 

%% programmer 
%% Xiaobai Sun 
%% Last revision: Fall 2024 
%% 