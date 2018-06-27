function [Acc,iter,Alpha,obj] = MyARTL(Xs,Ys,Xt,Yt,options)
% Adaptation Regularization
% Inputs:
%%% Xs  :   source feature matrix, ns * m
%%% Ys  :   source label vector, ns * 1
%%% Xt  :   target feature matrix, nt * m
%%% Yt  :   target label vector, nt * 1
%%% options:option struct
% Outputs:
%%% Acc    :   final accuracy using knn, float
%%% iter   :   list of all accuracies during iterations
%%% Alpha  :   coefficient matrix
%%% obj    :   objective value of f

%% Load algorithm options
p = options.n_neighbor;      % number of neighbors in algorithm (p)
sigma = options.sigma;       % refer to the paper
lambda = options.lambda;     % refer to the paper
gamma = options.gamma;       % refer to the paper
kernel_type = options.kernel_type;   % Kernel type: rbf | linear
T = options.T;               % number of iteration

fprintf('Algorithm ARTL started...\n');
%% Set predefined variables
Xs = Xs';
Xt = Xt';
X = [Xs,Xt];
Y = [Ys;Yt];
n = size(Xs,2);
m = size(Xt,2);
nm = n+m;
E = diag(sparse([ones(n,1);zeros(m,1)]));
YY = [];
for c = reshape(unique(Y),1,length(unique(Y)))
    YY = [YY,Y==c];
end
[~,Y] = max(YY,[],2);

%% Data normalization
X = X*diag(sparse(1./sqrt(sum(X.^2))));

%% Construct graph Laplacian
if gamma 
    manifold.k = p;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
    W = lapgraph(X',manifold);
    Dw = diag(sparse(sqrt(1./sum(W))));
    L = speye(nm)-Dw*W*Dw;
else
    L = 0;
end

% Generate pseudo labels for the target domain
if ~isfield(options,'Yt0')    
% base classifier for target, default is 1NN
% can be changed by following codes

% Neural network
%     model = train(Y(1:n),sparse(X(:,1:n)'),'-s 0 -c 1 -q 1');
%     [Cls,~] = predict(Y(n+1:end),sparse(X(:,n+1:end)'),model);
% SVM
%     model = svmtrain(Y(1:n),X(:,1:n)','-c 10');
%     [Cls,~,~] = svmpredict(Y(n+1:end),X(:,n+1:end)',model);
% 1NN
    knn_model = fitcknn(sparse(X(:,1:n)'),Y(1:n),'NumNeighbors',1);
    Cls = knn_model.predict(sparse(X(:,n+1:end)'));
%random
%     Cls = randi(10,length(Yt),1);
else
    Cls = options.Yt0;
end

Acc = 0;
Alpha = 0;
iter = [];
for t = 1:T
    %% Construct MMD matrix
    e = [1/n*ones(n,1);-1/m*ones(m,1)];
    M = e*e'*length(unique(Y(1:n)));
    N = 0;
    for c = reshape(unique(Y(1:n)),1,length(unique(Y(1:n))))
        e = zeros(n+m,1);
        e(Y(1:n)==c) = 1/length(find(Y(1:n)==c));
        e(n+find(Cls==c)) = -1/length(find(Cls==c));
        e(isinf(e)) = 0;
        N = N + e*e';
    end
    M = M + N;
    M = M/norm(M,'fro');

    %% Adaptation Regularization based Transfer Learning
	K = kernel_artl(kernel_type,X,sqrt(sum(sum(X.^2).^0.5)/nm));
	Alpha = ((E+lambda*M+gamma*L)*K+sigma*speye(nm,nm))\(E*YY);
	F = K*Alpha;
    
    [~,Cls] = max(F,[],2);
    norms = norm((Y' - Cls') * E,'fro');
    obj = norms * norms + sigma * trace(Alpha' * K * Alpha) + trace(Alpha' * K * (lambda * M + gamma * L)* K * Alpha);

    %% Compute accuracy
    Acc = numel(find(Cls(n+1:end)==Y(n+1:end)))/m;
    Cls = Cls(n+1:end);
    iter = [iter;Acc];
    fprintf('Iteration:[%02d]>>Acc=%f,obj:%f\n',t,Acc,obj);

end

end

function K = kernel_artl(ker,X,sigma)

    switch ker
        case 'linear'

            K = X' * X;

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        

        case 'sam'

            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));

        otherwise
            error(['Unsupported kernel ' ker])
    end
end


