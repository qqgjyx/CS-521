function [ v ] = stepsGraph( A, k, u, Gtype  )
%  stepsGraph, given an ajacency matrix A without negative calues 
%   and an initial node vector u, returns the node vector v 
%   after k(>=1) steps of traversal on the graph. 
%   
%  Calling sequence : 
%      [ v ] = stepsGraph( A, k, u, 'walk' ) ; 
%      [ v ] = stepsGraph( A, k, u, 'transition' ) ; 
% 
%  Case 'walk'. u is the vector of departure nodes and v is the vector of arrival 
%      nodes after k-steps walking on the graph. 
%         
%  Case 2. A is a transition matrix with nonzero elements and unit column
%  sums (column stochastic), u is a nonzero vector with unit sum (a probability distribution). 
%  Then v is the probabilty distribution after k-step transitions. 
% 

% ----------------------------------------------------
% Xiaobai Sun, Duke CS 
% for the Numerical Analysis Class, Fall 2011 
% --------------------------------------------------

disp( '    in stepsGraph : ' ); 

switch Gtype 
    
    case 'walk' 
    A1 =  ( A ~= 0 );          % replace nonzero weights with 1s 
    A1 =  double( A1 ) ; 
    u0 =  ( u ~= 0 ) ;
    disp( sprintf('     #departure-nodes = %d', nnz(u0) )); 
    u0 =  double( u0 ); 
    
    case 'transition' 
    D = sum(A,1);             
    A1 = A*diag(1./D);        % A1 is the right/column transition matrix 
    u0 = u/sum(u); 
    
    otherwise 
        error( 'stepsGraph : unknown traversal type' ) ; 
end

v = u0; 
for j = 1: k                  % take k steps 
    v = A1 * v; 
end

if strcmp( Gtype, 'walk' ) 
    v = ( v~= 0 );
    disp( sprintf('     #arrival-nodes = %d', nnz(v) ));     
end

%    ... visual display 

flagPlot = 0; 

if flagPlot 
    n = size(v); 

    figure( gcf ); 
    
    subplot(1,2,1) 
    plot( 1:n, u0, 'm.');
    xlabel( 'node index' ); 
    title( 'The intial ' ) 
    
    subplot(1,2,2) 
    plot( 1:n, v, 'm.' ) ;
    xlabel( 'node index' ) 
    ylabel( 'node function' ); 
    title( sprintf( ' after %d Steps',k));  
end

return

