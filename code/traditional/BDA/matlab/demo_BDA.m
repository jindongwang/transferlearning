%% Choose domains from Office+Caltech
%%% 'Caltech10', 'amazon', 'webcam', 'dslr' 
src = 'caltech.mat';
tgt = 'amazon.mat';

%% Load data
load(['data/' src]);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);    clear fts
Ys = labels;           clear labels

load(['data/' tgt]);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);     clear fts
Yt = labels;            clear labels

%% Set algorithm options
options.gamma = 1.0;
options.lambda = 0.1;
options.kernel_type = 'linear';
options.T = 10;
options.dim = 100;
options.mu = 0;
options.mode = 'W-BDA';
%% Run algorithm
[Acc,acc_ite,~] = BDA(Xs,Ys,Xt,Yt,options);
fprintf('Acc:%.2f',Acc);
