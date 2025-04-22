function h = gplot3 (A, xyz, varargin)
% 
% GPLOT3 - 3D graph plot
%
% SYNTAX
%
%   GPLOT3( A, XYZ, <LineSpec> )
%   H = GPLOT3( A, XYZ, <LineSpec> )
%
% INPUT
%
%   A           Graph adjacency matrix          [N-by-N; sparse]
%               (undirected & acyclic)
%   XYZ         3D node coordinates             [N-by-3]
%   <LineSpec>  LineSpec options for plot3      [varargin]
%
% OUTPUT
%
%   H           PLOT3 axes handle               [handle]
%
% DESCRIPTION
%
%   GPLOT3(A,XYZ,<LineSpec>) plots a 3D graph, much like the
%   built-in GPLOT function for 2D graphs.
%
%
% See also      gplot, plot3
%
    
    
    % source- and target-nodes for each edge in A
    [i, j] = find( tril(A) );
    X = [xyz(i,1), xyz(j,1)].';
    Y = [xyz(i,2), xyz(j,2)].';
    Z = [xyz(i,3), xyz(j,3)].';
    
    % graph plotting
    if nargout > 0
        h = plot3( X, Y, Z, varargin{:} );
    else
        plot3( X, Y, Z, varargin{:} );
    end
    
    
end



%%------------------------------------------------------------
%
% AUTHORS
%
%   Xiaobai Sun                        xiaobai@cs.duke.edu
%   Alexandros-Stavros Iliopoulos       ailiop@cs.duke.edu
%
% VERSION
%
%   1.1 - July 24, 2015
%
% CHANGELOG
%
%   1.1 (Jul 24, 2015) - Alexandros
%       * only lower-triangular part of adjacency matrix is used to
%         reduce unnecessary strain on the MATLAB renderer due to
%         double-plotting each edge; this also removes self-loops
%       * LineSpec is now passed verbatim to underlying PLOT3 call
%         (previous handling of LineSpec appears to be deprecated)
%       * removal of NaN-appended output option
%
%   1.0 (Nov 01, 2011) - Xiaobai
%       * initial implementation, for use in Duke CS Numerical
%         Analysis class
%
% ------------------------------------------------------------
