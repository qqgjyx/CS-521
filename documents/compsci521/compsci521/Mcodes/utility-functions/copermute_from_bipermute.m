
function [pBrows, pBcols] = copermute_from_bipermute( Bsizes, Bsubrows, Bsubcols, pAsub )
% 
%  [pBrows, pBcols] = ... 
%      copermute_from_bipermute( Bsizes, Bsubrows, Bsubcols, pAsub ) ;
%   
%  f.g. 
%  [prows, pcols] = copermute_from_bipermute( [m,n], 1:m, 1:n, ramdperm(m+n));
%  in the case Bsub = B 
% 
%  renders row permuation and column permutation of matrix B, 
%  according to a co-permutation of a submatrix Bsub via 
%  a bi-permutation in its symmetric embedding 
%          Asub = [ 0, Bsub; Bsub', 0 ] 
%
%   INPUT 
%   - Bsizes: 1x2 integer array, 
%                Bsizes(1) = size(B,1) = nrB; 
%                Bsizes(2) = size(B,2) = ncB;
%   - Bsubrows: nrBsub x1 integer arrays , nrBsub <= nrB 
%   - Bsubcols: ncBsub x1 integer arrays , ncBsub <= ncB 
%               indices into submatrix Bsub  
%  
%   - pAsub: (nr+nc)x1 integer array 
%            permtation of Asub 
%     
%  OUTPUT 
%   - pBrows Bsizes(1)x1: row permutation of B 
%   - pBcols Bsizes(2)x1: column permuation of B 
%
%   See also bipartite.symmetrization
%

% arguments
%   Bsizes    (2,1) {mustBeInteger}
%   pAsub     (:,1) {mustBeInteger}
%   Bsubrows  (:,1) {mustBeInteger}
%   Bsubcols  (:,1) {mustBeInteger}
% end

nrB = Bsizes(1);
ncB = Bsizes(2);

nrBsub = length( Bsubrows );
ncBsub = length( Bsubcols );

%  ... set the markers for bipartite-embedding of Bsub: 1 for rows; 2 for columns 
bimarker = [ ones(nrBsub,1); 2*ones(ncBsub,1)];
bimarker = bimarker( pAsub );

% ... separate row and column indices in the bi-permutation 
pr = pAsub( bimarker == 1 );
pc = pAsub( bimarker == 2 ) - nrBsub ;

% ... permute the given indices at input 
prBsub = Bsubrows(pr);
pcBsub = Bsubcols(pc);

% ... render co-permutation in B: place Bsub first, the remaining to the end
pBrows = [ prBsub; setdiff( ( 1:nrB )' , prBsub ) ];
pBcols = [ pcBsub; setdiff( ( 1:ncB)' , pcBsub ) ];

end

% Revision of recover_nonsymmetric_perm.m by 
% Dimitris F. <dimitrios.floros@duke.edu> 
% 
% Revision by Xiaobai Sun
% all variables renamed to be self-evident + additional document 
% Nov. 22, 2024 
% 
