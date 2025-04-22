function [A,gname] = binomialtree(q)
% [A,gname] = binomialtree(q); 
% generates the adjacency matrix of the binomial tree 
% with  n = 2^q nodes, q>-0 
%

% q = floor(log2(n));

A    = rec_subtree(q);

gname = sprintf('binomialTree-%d',q);

% plot(graph(A),'Layout','force','MarkerSize',4)

end

% -------------------------------------------------
function A = rec_subtree(q)

if q == 0
  A = 0;
else
  A0 = rec_subtree(q-1);
  
  Z  = sparse( 2^(q-1), 2^(q-1)); 
  Z(1,1) = 1;
  
  A  = [A0 Z; Z A0];

end

end

%% ===============
%%  Programmer 
%%  Nikos Pitsianis 
%%  July 2024 
%%  A base for the binomial options pricing model (BOPM) 
%%  assuming two possible prices, one up and one down;
%%  no dividends; constant interest rate ; no taxes or transition cost.
%% 
%%  initially for testing a graph compression algorithm 
%% 
