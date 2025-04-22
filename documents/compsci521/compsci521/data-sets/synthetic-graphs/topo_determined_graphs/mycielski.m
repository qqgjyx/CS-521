function [M,str, L, X] = mycielski( k )
  % 
  %  M = mycielski(k); 
  % 
  %  k: recursion depth/order 
  %  k+2: the chromatic number 
  %  symemtric, triangle-free  
  %  k+1 vertex connected 
  % 
  %  Generate Mycielsky graph/matrix of order k, the recursion depth 
  %    mu^k (G) := mu( mu^(k-1) (G) ),      k >= 0 
  % 
  %  mycielski(0) is the single edge graph K2
  %  mycielski(1) is the cycle-5 C5 = mu(K2) 
  %  mycielski(2) is the Grötzsch graph, n = 11, m = 20, mu^2(K2) 
  %  In general, 
  %  n = size( M_k, 1 ) = 3x2^(k)-1 ;         % mu^k(K2), n is odd  
  %  m = nnz( M_k )/2   follows the integer sequence A122695 in the OEIS 
  %  
  
  %% ... initialization clique number
  
  M = [0 1; 1 0];  % two-nodes, single edge 
  
  %% ... recursion 
  
  for i = 1: k
    M = Mski_recursion(M);
  end
  
  str = sprintf('Mycielski(%d)', k);
  L   = ones(size(M,1), 1);
  X   = [];

  end
  
  %%%% 
  
  function nuM = Mski_recursion(M)                                                   
  
  n = size(M,1);
  m = nnz(M);     % 2|E(G)| 
  
  n2 = 2*n+1;     % (n+1) new vertices, 2m+n new edges 
  m2 = 3*m+2*n;   % M copied as a subgraph and 
                  %          as a bipartite btw nodes(1:n) and nodes(n+(1:n)) 
                  % + K_{n,1} with the start center at node (2n+1) 
                  % m2 = 2|E(nuG)|  
  
  nuM = sparse([], [], [], n2, n2, m2 ); % in non-symmetric sparse format 
  
  nuM( 1:n, 1:n )  = M;   % copy the old to the leading matrix 
  nuM( n+[1:n], :) = sparse([M  sparse(n,n)  ones(n,1)]) ;
  nuM( :, n+[1:n]) = sparse([M; sparse(n,n); ones(1,n)]) ;
  
  % node (2n+1) is the star center, to leaf nodes n+(1:n) 
  end                                                                             
  
  %% Programmer 
  %%  Nikos Pitsianis 
  %% 
  %% Documentation by Xiaobai Sun 
  %% 
  %% Properties : nu(M) Mycielski, Jan (1955) 
  %% If M is triangle free, nuM is triangle free 
  %% chromatic-number(nuM)   = 1 + chromatic-numer(M)     
  %% domination-number(nuM ) = 1 + domination-number(M) 
  %% clique-number( nuM )    = max( 2, clique-number (M) ) 
  %% M has Hamiltonian cycle, so does nuM 
  %% M is factor critical, so is nuM 
  %% 
  %% See also Generalized Mycielski by Stiebitz, M. (1985)
  %% Mycielskians and Matchings, by Tomislav Doslic (2005) 
  %% 