function [ pfiedler, idxcut, fpm ]  = locate_the_cut( A, vfiedler )
% 
%  [ pfiedler, idx_cut, fpm ]  = locate_the_cut( A, vfiedler ); 
% 
% INPUT 
% A: nxn matrix, non-negative, symmetric for a connected, undirected graph 
% vfiedler: nx1 for the Fiedler vector of A's Laplacian 
% 
% OUTPUT 
% pfiedler: nx1 permutation, by sorting vfiedler 
% idxcut:   integer between 1 and n-1 
% fpm:      the sign of the largest element of vfiedler in magnitude
% 

v2 = vfiedler; 
[~, imax ] = max( abs(v2) );        % locate the dominant one 
fpm = sign( v2(imax) ); 
v2  = fpm * v2 ;                    % make the dominant one positive 
                                    % to eliminate sign flips in ordering 

[ v2p, pfiedler] = sort( v2, 'ascend'); 

Ap = A( pfiedler, pfiedler) ;
n  = size( A, 1);
cutscores = zeros(n-1,1);

for j = 1:n-1        % prototype, can be implemented efficiently 
                     % in practice, local to the zero crossing of v2p 

    alphaj = sum( Ap(1:j, 1:j),      'all' );  % leading 
    betaj  = sum( Ap( j+1:n, j+1:n), 'all' );  % trailing 
    gammaj = sum( Ap(1:j, j+1:n),    'all' );  % intersection 

    cutscores(j) = gammaj *( 1./alphaj + 1./betaj ) ;   
                 % normalize by the Harmonic mean of alphaj and betaj 
    
end
[ ~, idxcut ] = min( cutscores );

%% ... display 

cutscores    = sqrt(cutscores);
v2psqrt_pm   = sign(v2p) .* sqrt( abs(v2p)); 

figure 
n = length(v2); 
plot( 1:n-1, cutscores , 'm-.', 1:n, v2p, 'g+');
hold on
plot( idxcut, cutscores(idxcut), 'r*', 1:n, zeros(1,n), 'k--', 1, -1, 'k.');
title('cutscores along Fiedler ordering')



fprintf('\n      the suggested cut is at %d among (1:%d)', idxcut, n);

end

% Advanced notes: 
% Experimental finding with test data 
% (consistent with theoretical understanding) 
% 
% (1) the curve with a mean of negative p < -1 (Harmonic) gets more flat 
%     and hard to locate a minimum 
%  p = -2
%  cutscores(j) = gammaj *( 1./alphaj^2 + 1./betaj^2 )^(1/2)
% 
% (2) the curve with a mean of power p >= -1 gets more oscillatory 
%  p =1/2 
%  cutscores(j) = gammaj *( 1./alphaj^(1/2) + 1./betaj^(1/2) )^2 ;
%                 the curve is less oscillatory than p=0 the geometric mean 
% p = 0 
% cutscores(j) = gammaj / sqrt( alphaj * betaj );  
%                 with the geometric mean, the curve has more local minima 
% p=1 
% cutscores(j) = gammaj /( alphaj + betaj );  
%                 with the arithmetic mean, the curve has more minima near 
%                 the boundaies (extrme cuts) 
% 

%% Normal Cuts 
%% Reference: Shi and Malik, 1999-2000, on normal cuts and image segmentation
%% 
%% Programmer 
%% Xiaobai Sun 
%% 