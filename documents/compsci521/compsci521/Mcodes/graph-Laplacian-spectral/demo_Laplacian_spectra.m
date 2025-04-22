% function  demo_Laplacian_spectra 

close all
clear all

fprintf('\n   %s begin \n\n', mfilename ); 

demo_default = 0; %% input('   default demo [1,0] = '); 

if demo_default 
 gname = 'buckyball(C60): '; 
 A = bucky();  
else 
 addpath('../data-sets/synthetic-graphs/');
 select_topological_graphs 
 % select_random_graphs 
end

%%% =============================================

bnormalized = 1; 

n = size(A,1);
if n < 3500 
  Leig = Laplacian_spectra_full( A, bnormalized ) ; 
else 
 fprintf('\n   use the sparse solver for large graph/matrix ');
 return 
end

% ... the eigenvalue curve 

figure 
plot( Leig.S, 'bx' );
title( [gname, 'Laplacian eigenvalues' ] ) ;
xlabel('eigen-mode index'); 
ylabel('eigenvalues');

% --- the eigenvector matrix 

figure 
imagesc(Leig.V);
axis image 
title( [gname, 'Laplacian eigenvvectors' ] ) ;
xlabel('eigen-mode index'); 
ylabel('eigenvector/column ');

% ... spectral embedding in 2D, 3D 

xyzidx = [2,3,4]; 

fprintf('\n   2D vertex embeddng with low-energy modes ...')
figure 
plot( graph(A), 'XData', Leig.V(:,xyzidx(1)), ... 
                'YData', Leig.V(:,xyzidx(2)) ); 
axis equal; box on ; grid on; 
xlabel(sprintf('mode %d', xyzidx(1)) ) ; 
ylabel(sprintf('mode %d', xyzidx(2)) ) ; 
title( [gname, 'Laplacian Embedding2D'] ) ;

fprintf('\n   3D vertex embeddng with low-energy modes ...')
figure 
plot( graph(A), 'XData', Leig.V(:,xyzidx(1)), ... 
                'YData', Leig.V(:,xyzidx(2)), ... 
                'ZData', Leig.V(:,xyzidx(3)) ); 
axis equal; box on ; grid on; rotate3d;
xlabel(sprintf('mode %d', xyzidx(1)) ) ; 
ylabel(sprintf('mode %d', xyzidx(2)) ) ; 
zlabel(sprintf('mode %d', xyzidx(3)) ) ; 
title( [gname, 'Laplacian Embedding3D'] ) ;

fprintf('\n   3D vertex embeddng with high-energy modes ...')
figure 
n = size(A,1);
xyzidx = n-(0:2);
plot( graph(A),  ...
                'XData', Leig.V(:,xyzidx(1)), ... 
                'YData', Leig.V(:,xyzidx(2)), ... 
                'ZData', Leig.V(:,xyzidx(3)) ); 
axis equal; box on ; grid on; rotate3d ;
xlabel(sprintf('mode %d', xyzidx(1)) ) ; 
ylabel(sprintf('mode %d', xyzidx(2)) ) ; 
zlabel(sprintf('mode %d', xyzidx(3)) ) ; 
title( [gname, 'Laplacian Embedding3D' ] ) ;

fprintf('\n\n   %s end \n\n', mfilename );  

return 

% Often, FORCE-based embedding seems better, 
% although without freedom in reference frame selection 
% 
% figure 
% plot( graph(A), 'Layout', 'force3' );
% axis equal; box on ; grid on; rotate3d ; 

%% ====================
%% programmer 
%% Xiaobai Sun 
%% Duke CS 
%% Last revision: Sept. 1, 2024 
%% 
