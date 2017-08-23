function [acc,acc_ite,Alpha] = MyARTL(X_src,Y_src,X_tar,Y_tar,options)
    % Inputs:
    %%% X_src  :source feature matrix, ns * m
    %%% Y_src  :source label vector, ns * 1
    %%% X_tar  :target feature matrix, nt * m
    %%% Y_tar  :target label vector, nt * 1
    %%% options:option struct
    % Outputs:
    %%% acc    :final accuracy using knn, float
    %%% acc_ite:list of all accuracies during iterations
    %%% A      :final adaptation matrix, (ns + nt) * (ns + nt)
    
	%% Set options
	lambda = options.lambda;              %% lambda for the regularization
	kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
	T = options.T;                        %% iteration number
    n_neighbor = options.n_neighbor;
    sigma = options.sigma;
    gamma = options.gamma; 
    
    X = [X_src',X_tar'];
    Y = [Y_src;Y_tar];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	ns = size(X_src,1);
	nt = size(X_tar,1);
    nm = ns + nt;
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Y_src));
    E = diag(sparse([ones(ns,1);zeros(nt,1)]));
    YY = [];
    for c = reshape(unique(Y),1,length(unique(Y)))
        YY = [YY,Y==c];
    end
    
    %% Construct graph laplacian
    manifold.k = options.n_neighbor;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
    [W,Dw,L] = construct_lapgraph(X',manifold);
	%%% M0
	M = e * e' * C;  %multiply C for better normalization

    acc_ite = [];
    Y_tar_pseudo = [];
    % If want to include conditional distribution in iteration 1, then open
    % this
    
%     if ~isfield(options,'Yt0')
% %         model = train(Y(1:ns),sparse(X(:,1:ns)'),'-s 0 -c 1 -q 1');
% %         [Y_tar_pseudo,~] = predict(Y(ns+1:end),sparse(X(:,ns+1:end)'),model);
%         knn_model = fitcknn(X_src,Y_src,'NumNeighbors',1);
%         Y_tar_pseudo = knn_model.predict(X_tar);
%     else
%         Y_tar_pseudo = options.Yt0;
%     end

	%% Iteration
	for i = 1 : T
        %%% Mc
        N = 0;
        if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(nm,1);
                e(Y_src==c) = 1 / length(find(Y_src==c));
                e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end

        M = M + N;
        M = M / norm(M,'fro');
        
        %% Calculation
        K = kernel_artl(kernel_type,X,sqrt(sum(sum(X.^2).^0.5)/nm));
        Alpha = ((E + lambda * M + gamma * L) * K + sigma * speye(nm,nm)) \ (E * YY);
        F = K * Alpha;
        [~,Cls] = max(F,[],2);

        Acc = numel(find(Cls(ns+1:end)==Y(ns+1:end)))/nt;
        Y_tar_pseudo = Cls(ns+1:end);
        fprintf('Iteration [%2d]:ARTL=%0.4f\n',i,Acc);
        acc_ite = [acc_ite;Acc];
	end

end

function [W,Dw,L] = construct_lapgraph(X,options)
    W = lapgraph(X,options);
    Dw = diag(sparse(sqrt(1./sum(W))));
    L = speye(size(X,1)) - Dw * W * Dw;
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