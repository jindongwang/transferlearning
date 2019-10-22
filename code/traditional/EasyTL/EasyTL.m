function [acc,y_pred] = EasyTL(Xs,Ys,Xt,Yt,intra_align,dist,lp)
% Easy Transfer Learning

% Inputs:
%%% Xs          : source data, ns * m
%%% Ys          : source label, ns * 1
%%% Xt          : target data, nt * m
%%% Yt          : target label, nt * 1
%%%%%% The following inputs are not necessary
%%% intra_align : intra-domain alignment: coral(default)|gfk|pca|raw
%%% dist        : distance: Euclidean(default)|ma(Mahalanobis)|cosine|rbf
%%% lp          : linear(default)|binary

% Outputs:
%%% acc         : final accuracy
%%% y_pred      : predictions for target domain
    
% Reference:
% Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang.
% Easy Transfer Learning By Exploiting Intra-domain Structures.
% IEEE International Conference on Multimedia & Expo (ICME) 2019.

    C = length(unique(Ys));                 % num of shared class
    if C > max(Ys)
        Ys = Ys + 1;
        Yt = Yt + 1;
    end
    m = length(Yt);         % num of target domain sample
    if nargin == 4
        intra_align = 'coral';
        dist = 'euclidean';
        lp = 'linear';
    end
    if nargin == 5
        dist = 'euclidean';
        lp = 'linear';
    end
    if nargin == 6
        lp = 'linear';
    end
    switch(intra_align)
        case 'raw'
            fprintf('EasyTL using raw features...\n');
        case 'pca'
            fprintf('EasyTL using PCA...\n');
            dim = getGFKDim(Xs,Xt);
            X = [Xs;Xt];
            [~,score] = pca(X);
            X_new = score(:,1:dim);
            Xs = X_new(1:length(Ys),:);
            Xt = X_new(length(Ys) + 1 : end,:);
        case 'gfk'
            fprintf('EasyTL using GFK...\n');
            [Xs,Xt,~] = GFK_map(Xs,Ys,Xt,Yt,dim);
        case 'coral'
            fprintf('EasyTL using CORAL...\n');
            Xs = CORAL_map(Xs,Xt);
    end
    [~,Dct]= get_class_center(Xs,Ys,Xt,dist);
    fprintf('Start intra-domain programming...\n');
    [Mcj] = label_prop(C,m,Dct,lp);
    [~,y_pred] = max(Mcj,[],2);
    acc = mean(y_pred == Yt);
end

function [source_class_center,Dct] = get_class_center(Xs,Ys,Xt,dist)
% Get source class center and Dct
    source_class_center = [];
    Dct = [];
    class_set = unique(Ys)';
    for i = class_set
        indx = Ys == i;
        X_i = Xs(indx,:);
        mean_i = mean(X_i);
        source_class_center = [source_class_center,mean_i'];
        switch(dist)
            case 'ma'
                Dct_c = mahal(X_i,Xt);
            case 'euclidean'
                Dct_c = sqrt(sum((mean_i - Xt).^2,2));
            case 'sqeuc'
                Dct_c = sum((mean_i - Xt).^2,2);
            case 'cosine'
                Dct_c = cosine_dist(Xt,mean_i);
            case 'rbf'
                Dct_c = sum((mean_i - Xt).^2,2);
                Dct_c = exp(-Dct_c / 1);
        end
        Dct = [Dct,Dct_c];
    end
end

function [Dist] = sample_pair_distance(X)
    [n_sample,dim] = size(X);
    Dist = zeros(n_sample,n_sample);
    for i = 1 : n_sample
        sample_i = X(i,:);
        for j = 1 : n_sample
            sample_j = X(j,:);
            dist_ij = sum((sample_j - sample_i).^2);
            Dist(i,j) = dist_ij;
        end
    end
end

function D = cosine_dist(A,B)
% {cosine} computes the cosine distance.
%
%      D = cosine(A,B)
%      
%      A: M-by-P matrix of M P-dimensional vectors 
%      B: N-by-P matrix of M P-dimensional vectors
% 
%      D: M-by-N distance matrix
%
% Author: Stefano Melacci (2009)
%         mela@dii.unisi.it
%         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

    if (size(A,2) ~= size(B,2))
        error('A and B must be of the same dimensionality.');
    end

    if (size(A,2) == 1) % if dim = 1...
        A = [A, zeros(size(A,1),1)];
        B = [B, zeros(size(B,1),1)];
    end

    aa=sum(A.*A,2);
    bb=sum(B.*B,2);
    ab=A*B';

    % to avoid NaN for zero norms
    aa((aa==0))=1; 
    bb((bb==0))=1;

    D = real(ones(size(A,1),size(B,1))-(1./sqrt(kron(aa,bb'))).*ab);
end