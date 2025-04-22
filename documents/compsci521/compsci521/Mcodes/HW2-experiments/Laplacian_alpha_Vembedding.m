function [ piFiedler, idxcut, Y ]  = Laplacian_alpha_Vembedding( A, alpha, Lcolors, dimY) 
%
%  [ piFiedler, idxcut, v3 ]  = ... 
%             Laplacian_alpha_Vembedding( A, alpha, Lcolors ) ;
% 
% INPUT 
% A:      nxn matrix, non-negative,  representing a graph 
% alpha:  real or complex valued, alpha \neq 1 if A is nonsymmetric 
% Lcolors: nx3 RGB colors for ground-truth  vertex labels 
%
% OUTPUT 
% piFiedler: the permutation with Fiedler elements in ascending order 
% idxcut:    the index to the normal cut location 
% v3:        the third eigenvector, elements in piFiedler ordering 

fprintf('\n\n    enter %s \n', mfilename ); 

if issymmetric(A) 
    fprintf('\n     matrix A is symmetric');
    alpha = 1;
end

n    = size(A,1);
din  = A  * ones( n, 1); 
dout = A' * ones( n, 1);

if min(din) <= 0 | min(dout) <= 0 
    error('there are zero rows or column)');
end

d     = din + alpha^2 * dout;
Ahat  = diag( d.^(-1/2) ) * (A+A') * diag( d.^(-1/2) ) ;  
Ahat  = full( Ahat ) ; 
Lhat  = eye( n ) + alpha * Ahat ;

[ V, Lambda]   = eig(  Lhat,  "vector");
[ Lambdap, p]  = sort( Lambda, 'descend');

if Lambdap(2) > 2-eps 
    error('the Fiedler value is numerically zero; \n    graph is numerically disconnected'); 
end
%% Ahat is symmetric, connected       --> Ahat is irreducible 
%% If in addition, Ahat is triangular --> each eigenvalue is simple 
if Lambdap(2) == Lambdap(3) 
    fprintf('The Fiedler value is not simple;\n   graph has symmetrical structure');
end

figure 
plot( Lambdap, 'b+');
title('Laplacian-alpha eigenvalues in descending order');

%% ... get the spectral encoding, or embedding coordinates, of the vertices , with eigenvalue scaling 
%%     it is the loading matrix in statistic sense 

Lambdap_sqrt = sqrt( Lambdap + eps );

X  = V(:,p) * diag( Lambdap_sqrt );  

%% ... get the permutation by the Fiedler vector 

[ piFiedler, idxcut, fpm ] = locate_the_cut( Ahat, X(:,2) );

Y = X( piFiedler, 1:dimY );

if n < 2500 
 figure 
 plot( 1:n, sqrt(d(piFiedler)), 'b+', 1:n, fpm*X( piFiedler,2) , 'g+');
 hold on 
 plot(1,-1,'k.');
 title('Perron vs (sorted) Fiedler');

 figure 
 subplot(1,3,1)
 imagesc(A, 'AlphaData', A > 0);
 n = size( X, 1);

 axis equal tight 
 colorbar
 colormap( gca, flipud(parula(5)) )
 xlabel('A at input') 
% 
 subplot(1,3,2) 
 imagesc(Ahat, 'AlphaData', Ahat > 0);
 axis equal tight 
 colorbar
 colormap( gca, flipud(parula(5)) )
 xlabel('Ahat (similar to walk transition)')
% 
 subplot(1,3,3) 
 imagesc( Ahat( piFiedler, piFiedler ), 'AlphaData', Ahat( piFiedler, piFiedler ) > 0);
 axis equal tight 
 xlabel('Ahat in Fielder order')
 colorbar 
 colormap( gca, flipud(parula(5)) )
end

%% ... label vec --> color vec   

figure  
scatter( X(:,1), X(:,2), 15, Lcolors(:,1:3), 'filled' ); 
xlabel(' Perron ')
ylabel(' Fiedler ')
title( [ 'L+ Vembedding (2D) ' ] ); 

figure 
scatter3( X(:,1), X(:,2), X(:,3), 15, Lcolors(:,1:3), 'filled' ); 
xlabel('Perron')
ylabel('Fiedler')
zlabel('v3')
title( [ 'L+ Vembedding (3D) '] );


figure 
scatter3( X(:,2), X(:,3), X(:,4), 15, Lcolors(:,1:3), 'filled' ); 
xlabel('v2( Fiedler ) ')
ylabel('v3')
zlabel('v4')
title( [ 'L+ Vembedding (3D) '] );


fprintf('\n\n    exit %s \n\n', mfilename ); 


return 

%% Programmer 
%% Xiaobai Sun 
%% Duke CS 
