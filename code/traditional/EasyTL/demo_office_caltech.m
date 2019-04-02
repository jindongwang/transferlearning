% Test with Office-Caltech datasets

str_domains = { 'Caltech10', 'amazon', 'webcam', 'dslr' };
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load(['data/office+caltech--surf/' src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts,1);    clear fts
        Ys = labels;           clear labels
        
        load(['data/office+caltech--surf/' tgt '_SURF_L10.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts,1);     clear fts
        Yt = labels;            clear labels
        
        % EasyTL without intra-domain alignment [EasyTL(c)]
        [Acc1, ~] = EasyTL(Xs,Ys,Xt,Yt,'raw');
        fprintf('Acc: %f\n',Acc1);
        
        % EasyTL with CORAL for intra-domain alignment
        [Acc2, ~] = EasyTL(Xs,Ys,Xt,Yt);
        fprintf('Acc: %f\n',Acc2);
        
        list_acc = [list_acc;[Acc1,Acc2]];
    end
end
