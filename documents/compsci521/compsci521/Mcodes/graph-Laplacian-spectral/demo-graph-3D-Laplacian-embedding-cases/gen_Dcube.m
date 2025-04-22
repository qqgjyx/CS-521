function A = genDcube( d, figID) 
% 
% A = genDcube( d, figID) ;   % d >= 1
% 
% generate  the adjacency matrix A for the d-dimensional 
% hypercube. 
% 
% when nargin > 1, display the generation process 
% 

% ---------------------------------------------------
% Xiaobai Sun 
% Duke CS 
% For Numerical Analysis Class
% Oct. 2011 
%  
% Contact xiaobai for any further distribution 
% ---------------------------------------------------



% ... the base : d = 1 

nv = 2 ;  
 
A = [0 1; 1 0];

    
% ... double the number of nodes every time 

if nargin == 1  % without displaying the generation process 
    
  for i = 2:d,
    A = [ A, speye( nv ); ... 
          speye( nv ), A ];
    nv = nv * 2; 
  end
  
else
    
  di = 1;
  
  figure( figID ) ;
  strPrefix = 'the adjacency matrix for '; 
  strSuffix = sprintf( ' the %d-dim hypercube', di); 
  spy( A ) ;
  title( [strPrefix, strSuffix] ); 
  
  
  for i = 2:d,
    A = [ A, speye( nv ); ... 
          speye( nv ), A ];
    
    di = di + 1; 
    nv = 2 * nv ; 
    disp( '   press a key to COPY and LINK ' ); 
    pause( );
    
    spy(A)
    axis ij 
    strSuffix = sprintf( ' the %d-dim hypercube', di); 
    title( [ strPrefix, strSuffix ] ) ; 
  end 
  
end

return 
