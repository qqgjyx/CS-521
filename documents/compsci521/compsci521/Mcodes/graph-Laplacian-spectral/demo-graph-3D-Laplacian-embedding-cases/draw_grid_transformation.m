function uvBox = draw_grid_transformation( A, T, xy  ) 
% 
% cystomized drawring for grid transformation 
% 
% A  : adjacency matrix 
% xy : node coordinates 
% T  : transform matrix 


uv = xy * T' ;          % coordinate transformation  

B = [ 0, 1 , 0, 1 ; 
      1, 0 , 1, 0 ;
      0, 1,  0, 1 ; 
      1, 0,  1, 0 ] ;
  
UVmin = min( uv );
UVmax = max( uv ); 
uvBox = [ UVmin(1), UVmin(2) ; 
          UVmax(1), UVmin(2) ; 
          UVmax(1), UVmax(2) ;
          UVmin(1), UVmax(2) ] ; 

figure                  % display the transformation 
gplot( A, uv, 'b' ); 
hold on 
gplot( B, uvBox, 'm'); 
axis off 

end

% ===============
% Xiaobai Sun 
% Duke CS 
% 