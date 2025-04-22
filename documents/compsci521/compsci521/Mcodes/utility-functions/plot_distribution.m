function plot_distribution( d, dname, nbins, figID )
%
% plot_distribution( d, dname, nbins, figID ); 
% 
% d: vector of nonegative integers 
%    such as degree sequence, local-cluster-coefs sequence, etc.. 
% 

d = d + 10*eps;

figure( figID )
%% 
subplot(2,2,1)
histogram(d, 'Normalization', 'probability', 'NumBins', nbins ); 
ylabel('p(x)=y(x)/sum(y)')
xlabel('x') 
%%
title( [dname] ) ;
%% 
subplot(2,2,2)  % NE plot 
histogram( d , 'Normalization', 'probability', 'NumBins', nbins, 'FaceAlpha', 0.3 ) 
set( gca, 'XScale', 'log'); 
xlabel('log')
ylabel('linear')
%% 
subplot(2,2,3) 
histogram( d, 'Normalization', 'probability', 'NumBins', nbins, 'FaceAlpha', 0.3 ) 
set( gca, 'YScale', 'log'); 
ylabel('log y ')
xlabel('x')
%% 
subplot(2,2,4)
histogram( d, 'Normalization', 'probability', 'NumBins', nbins, 'FaceColor', 'magenta', 'FaceAlpha', 0.3 ) 
set( gca, 'YScale', 'log', 'Xscale', 'log' ); 
ylabel('log')
xlabel('log')

end

%%% programmer 
%%% Xiaobai Sun 