%%% SCRIPT: get_iris_data.m 

% close all
% clear all

load fisheriris 
nsamples    = size( meas, 1);
kattributes = size( meas, 2);

fprintf('\n   loaded data IRIS: [#specimens, #attributes] = [%d, %d]', nsamples, kattributes ); 
point_data_name = 'IRIS'; 

fprintf('\n\n   the specicies classes:  '); 
uniq_species_lables = unique( species ); 
for i = 1 : length( uniq_species_lables) 
    fprintf( '\n   %s', uniq_species_lables{i} ); 
end

%% 50 specimens per class

fprintf('\n\n   the attributes:  ');
attributes = ["sepal length","sepal width", "petal length","petal width"]; 
%% length in cm 
for i = 1 : length( attributes ) 
    fprintf( '\n   %s', attributes(i) ); 
end

%% make color labels 

vc      = double( categorical(species) );    % labels --> numerical 
Lcolors = full( sparse(1:numel(vc),vc, ones(size(vc)), numel(vc), 3 ));

%% shuffle the data 

rng   = 5;
piris = randperm( size(meas,1 )); 
meas0 = meas;
meas  = meas( piris, :);
Lcolors = Lcolors( piris, : );


figure 
imagesc( full(meas) ); 
title('Feature data: IRIS')
xlabel('attributes')
ylabel('specimens/flowers')
colorbar

%%
fprintf('\n\n   data IRIS ready to use \n\n'); 

return 