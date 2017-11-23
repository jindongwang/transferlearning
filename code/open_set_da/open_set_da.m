%% Open Set Domain Adaptation
%% Load data
str_domains = {'caltech','amazon','webcam','dslr'};
addpath(genpath('data/'));
for i = 2 : 2
	for j = 3 : 3
		if i == j 
			continue;
        end
        fprintf('i=%d,j=%d\n',i,j);
        src = str_domains{i};
		tar = str_domains{j};
		load(['office+caltech--decaf/',src,'_decaf.mat']);
		feas = feas ./ repmat(sum(feas,2),1,size(feas,2)); 
        Xs = zscore(feas);     clear feas
        Ys = labels;           clear labels

        load(['office+caltech--decaf/' tar '_decaf.mat']);     % target domain
        feas = feas ./ repmat(sum(feas,2),1,size(feas,2)); 
        Xt = zscore(feas);      clear feas
        Yt = labels;            clear labels
        
        [acc,acc_list,ot_list] = ati(Xs,Ys,Xt,Yt);
    end
end

function [acc,acc_list,ot_list] = ati(Xs,Ys,Xt,Yt)
% Core function
    acc = 0;
    acc_list = [];
    ot_list = [];
    C = 10;                 % num of shared class
    m = length(Yt);         % num of target domain sample
    D = size(Xs,2);         % num of features
    itera = 10;
    for i = 1 : itera
        [S,Dct]= get_class_center(Xs,Ys,Xt);
        lambda = 0.5 * (max(Dct(:)) + min(Dct(:)));  % lambda
        T = Xt';
        [Xct,Ot] = get_label_binary(C,m,lambda,Dct);
        W = feature_map(Xct,S,T,C,m);
        Xs = (W * Xs')';
        
        %% This is not in paper
        %% If want to see the accuracy of step 1
        [v,lab] = max(Xct,[],2);
        y_pred = v .* lab;
        ind = y_pred ~= 0;
        y_pred_one = y_pred(ind);
        y_true = Yt(ind);
        acc = mean(y_pred_one == y_true);

        acc_list = [acc_list;acc];
        ot_list = [ot_list;sum(Ot)];
        fprintf('Iteration %d: acc=%.4f,Ot:%d\n',i,acc,sum(Ot));
    end
end

function [source_class_center,Dct] = get_class_center(Xs,Ys,Xt)
% Get source class center and Dct
    source_class_center = [];
    Dct = [];
    class_set = unique(Ys)';
    for i = class_set
        indx = Ys == i;
        X_i = Xs(indx,:);
        mean_i = mean(X_i);
        source_class_center = [source_class_center,mean_i'];
        Dct_c = sum((mean_i - Xt).^2,2);
        Dct = [Dct,Dct_c];
    end
end
