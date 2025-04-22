% function A_rnn = demo_RNN_graph( X, r, name_metric   )
% 
%  A_rnn = construct_rnn_graph( X, r, name_metric   ) ; 
% 
%  INPUT 
%  X: n x d array, Feature matrix, n feature vector of length d 
%  r: real-valued, positive  
%  name_matric: character string for the bane of a metric function 
% 
%  OUTPUT 
%  A_rnn : nxn real-valued, nonnegative array 
%        as the adjacency matrix of a rNN graph 
%        using a default Gaussin kernel for conversion from 
%        pairwise discrepancy to pairwise dis-similarity  
%      
%  NOTE 
%     (1) neighbor information is stored in two cell arrays Idx_rnn, D_rnn 
%     (2) this demo SCRIPT can be turned easily into a FUNCTION as indicated
%     
%     
close all
clear all

fprintf('\n\n   %s began', mfilename);

% load fisheriris
get_iris_data            % including information and Lcolors 


X = meas ;               % Measurements of original flowers

%% ... range search parameter and metric options 

r = input( '\n   specify a radius = ') ;   % 1.5 
list_metrics = { 'chebychev', 'euclidean',  'mahalanobis' , 'minkowski' };
for  k = 1: length(list_metrics) 
    fprintf('\n   %d %s', k, list_metrics{k} ) ; 
end
idx_metric  = input('\n   choose a metric by the index = ');
name_metric = list_metrics{ idx_metric } ;

%% ... get the neighbors within distance r 

switch name_metric  
    case 'minkowski'    % norm(x-y, p)     
      pval = input('\n   input the power (p>=1) = '); 
      [Idx_rnn, D_rnn ]  =  rangesearch( X, X, r, 'Distance', name_metric, 'P', pval );  
    otherwise 
      [Idx_rnn, D_rnn ] =  rangesearch( X, X, r, 'Distance', name_metric ); 
end

% Note: when Y = X in range-search between X and Y,
%       D_rnn includes zero distance d(x,x) = 0

%% ... converting pairwise discrepancy to dis-similarity : cell-array to cell-array 

sigma = 3;

n = length( D_rnn ); 
fD_rnn = cell( n, 1 ); 
for i = 1:n
  fD_rnn{i} = exp( -D_rnn{i}.^2/sigma );
end 

%% ... convert to adjacency matrix ( in sparse format ) 

A_rnn = rnn2adjacency( Idx_rnn, fD_rnn  ); 
  

%% .. display the adjacency matrix 

 if size( Idx_rnn, 1) < 2500 
  figure 
  imagesc( A_rnn )
  axis equal tight 
  colorbar 
  % colormap( pink )
  tmsg = sprintf( 'RNN adjacency matrix (r = %g3.2)', r);
  title( tmsg ); 
 end 
 
[ piFiedler, idxcut, piv3 ] = Laplacian_Vembedding( A_rnn , 'flowers', Lcolors );

fprintf('\n   cut location at %d', idxcut ) ; 

figure 
imagesc( A_rnn(piFiedler, piFiedler)  );
title('Adjacency matrix in Fiedler ordering')
axis equal tight 
colorbar 

figure 
imagesc( X( piFiedler,:))
title('Feature data')
ylabel('Fiedler order')


figure 
imagesc( A_rnn( piv3,  piFiedler ) );
title('Adjacency matrix in two orderings') ;
xlabel('fiedler')
ylabel('spectral v3')
axis equal tight 
colorbar

fprintf('\n\n   %s finished \n\n\n', mfilename);

return 


%% programmer 
%% Xiaobai Sun 
%% Last revision: Fall 2024 
%% 
