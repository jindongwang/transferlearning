% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load(['data/' src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts,1);    clear fts
        Ys = labels;           clear labels
        
        load(['data/' tgt '_SURF_L10.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts,1);     clear fts
        Yt = labels;            clear labels
        
        % meda
        options.d = 20;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 10;
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
    end
end
