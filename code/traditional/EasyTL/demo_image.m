% Test image datasets (ImageCLEF or Office-Home) using resnet50 features

img_dataset = 'image-clef';  % 'image-clef' or 'office-home'

if strcmp(img_dataset,'image-clef')
    str_domains = {'c', 'i', 'p'};  
    addpath('Image-CLEF DA dataset path');  % You need to change this path
elseif strcmp(img_dataset,'office-home')
    str_domains = {'Art', 'Clipart', 'Product', 'RealWorld'}; 
    addpath('Office-Home dataset path');    % You need to change this path
end

list_acc = [];
for i = 1 : 3
    for j = 1 : 3
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        fprintf('%s - %s\n',src, tgt);

        data = load([src '_' src '.csv']);
        fts = data(1:end,1:end-1);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts, 1);
        Ys = data(1:end,end) + 1;
        
        data = load([src '_' tgt '.csv']);
        fts = data(1:end,1:end-1);
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts, 1);
        Yt = data(1:end,end) + 1;
        
        % EasyTL without intra-domain alignment [EasyTL(c)]
        [Acc1, ~] = EasyTL(Xs,Ys,Xt,Yt,'raw');
        fprintf('Acc: %f\n',Acc1);
        
        % EasyTL with CORAL for intra-domain alignment
        [Acc2, ~] = EasyTL(Xs,Ys,Xt,Yt);
        fprintf('Acc: %f\n',Acc2);
        
        list_acc = [list_acc;[Acc1,Acc2]];

    end
end
