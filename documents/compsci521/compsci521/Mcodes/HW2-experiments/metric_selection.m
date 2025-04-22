function [name_metric,p_mink_val] = metric_selection 
% 
%  name_metric = metric_selection ;
% 
%  select a metric from a list of options 

list_metrics = { 'chebychev', ... 
                 'euclidean', ...
                 'mahalanobis' , ... 
                 'minkowski' };    % add more 

for  i = 1: length(list_metrics) 
    fprintf('\n   %d %s', i, list_metrics{i} ) ; 
end

idx_metric  = input('\n   choose a metric by the index = ');
name_metric = list_metrics{ idx_metric } ;

p_mink_val = [];
switch name_metric
    case 'minkowski'
        p_mink_val = input('\n   input the power (p>=1) = ');    % p=5 is a charm 
    otherwise
end

end

%% Programmer 
%% Xiaobai Sun 
%% Duke CS 
