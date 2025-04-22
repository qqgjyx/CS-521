% function [A, gname]  = select_topological_graphs 
%  
% [A, gname]  = select_topological_graphs ;
%  A     : nxn array, for the graph adjacency matrix 
%  gname : charater string, for the graph name 

%% ... data collection 

close all 
clear all 

% addpath topo_determined_graphs

graph_data_files = {...
                    'binomialtree', 
                    'buckyball',  
                    'clique', 
                    'grid', 
                    'torus', 
                    'grapheneSheet',
                    'hypercube',
                    'mycielski',          
                    'rings', 
                    'rok', 
                    'wheel'
                    } ;

for q = 1:length( graph_data_files )
  fprintf('\n   %d : %s', q, graph_data_files{q} ) ; 
end
q = input('\n\n   enter the graph index = ');  

gname =  graph_data_files{q}; 

switch gname 
    case 'binomialtree'
        q = input('   binomialtree: #levels (>1) =  ');
        A = binomialtree(q);
    case 'buckyball'
        A = bucky();      % matlab built-in, n = 60 
    case 'clique' 
        n = input('   clique: #nodes = ');
        A = clique(n);
    case 'grid' 
        d = input('   grid: dimension = '); 
        ns = zeros(d,1); 
        for j = 1:d 
            nj_spec = sprintf('   dimension-%d length = ', j); 
            ns(j) = input( nj_spec );
        end 
        A = grid_and_torus(d, ns, 0);
    case 'torus' 
        d  = input('   torus: dimension = '); 
        ns = zeros(d,1); 
        for j = 1:d 
            nj_spec = sprintf('   dimension-%d length = ', j); 
            ns(j)     = input( nj_spec );
            btorus(j) = input('   circular [0.1] = ' ); 
        end 
        A = grid_and_torus( d, ns, btorus );

    case 'grapheneSheet' 
            ny = input('\n   Grapheen: nrows = '); 
            nx = input('\n             ncols = '); 
            A  = grapheneSheet(ny,nx); 

    case 'hypercube'
            d = input('   hypercube: dimension = '); 
            A = hypercube(d);

    case 'mycielski'
            k = input('   recursion level = '); 
            A = mycielski( k );

    case 'rings'
            n = input('   Rings: #nodes = ');
            k = input('   semi-bandwidth = ');
            A = rings(n,k);

    case 'rok'
            nk  = input('   RoK: clique size = ');
            mc  = input('   ring length (>2) = ');
            A   = rok( mc, nk);

    case 'wheel' 
           n = input('   Wheel: #nodes = ');
           A = wheel(n); 

    otherwise
        error('unknown graph generation case')
end

n = size(A,1);
m = ceil( nnz(A)/2 ); 

fprintf('\n\n   Graph %s:  [#nodes, #edges] = [%d, %d] \n\n', gname, n, m); 

return 

%% Programmer 
%% Xiaobai Sun 
%% Last revision: Aug. 31, 2024 
%% 