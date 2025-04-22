% function [A, gname]  = select_random_graphs 
%  
% [A, gname]   = select_random_graphs ;
%  A     : nxn array, for graph adjacency matrix 
%  gname : character string, for the graph name 

%% ... data collection 

addpath random-graphs 

graph_data_files = {...
                    'er_n3000_p0.3.mat',    
                    'ba_n3000_k20.mat',
                    'ws_n3000_b10_r0.mat',        % without rewiring 
                    'ws_n3000_b10_r0.01.mat',     % with rewiring 
                    'rg_n3000_d2_r0.0469_s0.mat', % random geometric 2D 
                                        } ;

for q = 1:length( graph_data_files )
  fprintf('\n   %d : %s', q, graph_data_files{q} ) ; 
end

q     = input('\n\n   enter the graph index = ');  
gname =  graph_data_files{q}; 

mf = load( [ graph_data_files{q} ]  ) ; 
A  = mf.A ;

n = size(A,1);

if issymmetric(A) 
   m = ceil( nnz(A - diag(diag(A) )) /2 );
else 
    m = nnz(A);
end

gname = gname(1:end-4); % remove file extension .mat

fprintf('\n   Graph %s : [n,m] = [%d, %d] \n\n', gname, n, m); 

return 

%% Programmer 
%% Xiaobai Sun 
%% last revision: Aug. 31, 2024 
%% 