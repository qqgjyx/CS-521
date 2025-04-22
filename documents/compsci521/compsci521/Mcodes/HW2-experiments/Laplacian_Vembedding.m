function [ piFiedler, idxcut, v3 ]  = Laplacian_Vembedding( A , v_names, Lcolors ) 
%
% Laplacian_Vembedding( A , var_names, Lcolors )  ; 
% 
% INPUT 
% A: nxn matrix, representing an undirected graph 
% v_names: character string for the vertex names 
% Lcolors: nx1 integer array, colors for ground-truth  vertex labels 
%
% OUTPUT 
% piFiedler: the permutation with Fiedler elements in ascending order 
% idxcut: the index to the cut location 
% v3: the third eigenvector, elements in piFiedler ordering 

fprintf('\n\n   %s began \n\n', mfilename ); 

if ~issymmetric(A) 
    A = ( A + A' )/2; 
end

n = size(A,1);

d = A * ones( n, 1); 
if min(d) <= 0 
    error('there are zero rows/column)');
end

Ahat  = diag( d.^(-1/2) ) * A * diag( d.^(-1/2) ) ;  


Ahat  = full( Ahat ) ; 
Lhat  = eye( n ) + Ahat ;

[ V, Lambda]   = eig(  Lhat,  "vector");
[ Lambdap, p]  = sort( Lambda, 'descend');

if Lambdap(2) > 2-eps 
    error('the graph is numerically disconnected'); 
end

figure 
plot( Lambdap, 'bo');
title('Laplacian+ eigenvalues in descending order');

%% ... get the spectral encoding, or embedding coordinates, of the vertices , with eigenvalue scaling 
%%     the loading matrix in statistic sense 

Lambdap_sqrt = sqrt( Lambdap + eps );

X  = V(:,p) * diag( Lambdap_sqrt );  
v2 = X(:,2);
[~, imax ] = max( abs(v2) );        % locate the dominant one
v2 = v2/v2(imax);                   % make the dominant positive  

%% ... get the permutation by the Fiedler vector 

[ piFiedler, idxcut ] = locate_the_cut( Ahat, v2 );

v3 = X(:,3);

if n < 2500 
 figure 

subplot(1,3,1) 
imagesc(A);
axis equal tight 
colorbar
xlabel('A') 
% 
subplot(1,3,2) 
imagesc(Ahat);
axis equal tight 
colorbar
xlabel('Ahat (walk)')
% 
subplot(1,3,3) 
 imagesc( Ahat( piFiedler, piFiedler ));
 axis equal tight 
 xlabel('Ahat in Fielder order')
 colorbar 
end

%% ... label vec --> color vec   

figure  
scatter( X(:,1), X(:,2), 15, Lcolors, 'filled' ); 
xlabel(' Perron ')
ylabel(' Fiedler ')
title( [ 'L+ Vembedding (2D) ' ] ); 

figure 
scatter3( X(:,1), X(:,2), X(:,3), 15, Lcolors, 'filled' ); 
xlabel('Perron')
ylabel('Fiedler')
zlabel('v3')
title( [ 'L+ Vembedding (3D) '] );


figure 
scatter3( X(:,2), X(:,3), X(:,4), 15, Lcolors, 'filled' ); 
xlabel('v2( Fiedler ) ')
ylabel('v3')
zlabel('v4')
title( [ 'L+ Vembedding (3D) '] );


fprintf('\n\n   %s finished \n\n', mfilename ); 


return 

%% Programmer 
%% Xiaobai Sun 
%% Duke CS 
