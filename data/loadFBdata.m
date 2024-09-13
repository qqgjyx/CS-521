%% Load SNAP Facebook data
% This script loads downloaded and unzipped files in 'facebook' directory in
% current folder into a bunch of workspace variables.

%% Getting Ready
% File names starts with the id of ego nodes. Let's start by extracting
% those node ids.

s = dir('facebook');                                % read 'facebook' dir
s(cat(1,s.isdir)) = [];                             % drop folders
s = s(arrayfun(@(x) x.name(1),s) ~= '.');           % drop OSX hidden files
s = arrayfun(@(x) strsplit(x.name,'.'), s, ...      % split file names
    'UniformOutput', false);
s = vertcat(s{:});                                  % unnest cell array
egoids = unique(s(:,1));                            % get ego node ids
egoids = cellfun(@(x) str2double(x), egoids);       % convert to double

%% Load Data
% We can now load the content of those files and create graph objects from
% the edges, which are the connections between nodes. Because the node ids
% are 0-indexed but MATLAB is 1-indexed, we need to increment the node ids
% by 1, but we can use 0-indexed node labels, because they are just
% strings. We also use undirected graphs because Facebook friendships are
% mutual.

circles = cell(size(egoids));                       % circle accumulator
edges = cell(size(egoids));                         % edge accumulator
feat = cell(size(egoids));                          % feat accumulator
egofeat = cell(size(egoids));                       % ego feat accumulatr
featnames = cell(size(egoids));                     % feat name accumulator
graphs = cell(size(egoids));                        % graph accumulator

for i = 1:length(egoids)                            % loop over ego node ids
    egoid = egoids(i);                              % get current id 
    f = ['facebook/',num2str(egoid)];               % filename
    
    fid = fopen([f,'.circles']);                    % open file
    cir = textscan(fid, ...                         % read file content
        ['%*s', repmat('%f',[1,310])], ....         % skip string, repeat float x 310
        'CollectOutput', true);                     % concatenate data
    cir = cir{1};                                   % unnest cell array
    cir(:,sum(isnan(cir)) == size(cir,1)) = [];     % remove NaN columns
    circles{i} = cir;                               % add to accumulator
    
    edgemat = dlmread([f,'.edges']);                % read edges
    edgemat = sort(edgemat, 2);                     % sort edge order
    edgemat = unique(edgemat,'rows');               % remove duplicates
    edges{i} = edgemat;                             % add to accumulator
    
    edgemat = edgemat + 1;                          % convert to 1-indexing
    G = graph(edgemat(:,1),edgemat(:,2));           % create undirected graph
    nids = 0:size(G.Nodes, 1) - 1;                  % restore 0-indexing
    nids = strtrim(cellstr(num2str(nids(:))));      % convert to cellstr
    G.Nodes = nids;                                 % add as node names
    graphs{i} = G;                                  % add to accumulator
    
    egofeat{i} = dlmread([f,'.egofeat']);           % add ego feat to accumulator
    feat{i} = dlmread([f,'.feat']);                 % add feat to accumulator
    
    T = readtable([f, '.featnames'], ...
        'FileType', 'text', ...
        'ReadVariableNames', false, ...
        'Delimiter', '\t', ...    % Adjust based on actual delimiter
        'Format', '%d %s');       % Use '%d' if feature indices are integers
    T.Properties.VariableNames = {'Id','Desc'};     % add variable names
    featnames{i} = T;                               % add to accumulator
end

clearvars -except circles edges feat egofeat featnames graphs egoids


% DFW
% ADJ
% JUN