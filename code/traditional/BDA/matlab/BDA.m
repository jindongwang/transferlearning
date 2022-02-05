function [acc,acc_ite,A] = BDA(X_src,Y_src,X_tar,Y_tar,options)
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
    
    % Reference:
    % Jindong Wang, Yiqiang Chen, Shuji Hao, and Zhiqi Shen.
    % Balanced distribution adaptation for transfer learning.
    % ICDM 2017.
    
	%% Set options
	if ~isfield(options,'mode')           %% 'BDA' or 'W-BDA'
        options.mode = 'W-BDA';
    end
    
    if ~isfield(options,'mu')             %% balance factor \mu
        options.mu = 1;
    end                 
    
    if ~isfield(options,'lambda')         %% lambda for the regularization
        options.lambda = 0.1;
    end
    
    if ~isfield(options,'dim')            %% dim is the dimension after adaptation, dim <= m
        options.dim = 10;
    end
    
    if ~isfield(options,'kernel_type')    %% kernel_type is the kernel name, primal|linear|rbf
        options.kernel_type = 'primal';
    end
    
    if ~isfield(options,'gamma')          %% gamma is the bandwidth of rbf kernel
        options.gamma = 1;
    end
    
    if ~isfield(options,'T')              %% iteration number
        options.T = 10;
    end
    
    mu = options.mu;
    lambda = options.lambda;
    dim = options.dim;
    kernel_type = options.kernel_type;
    gamma = options.gamma;
    T = options.T;

    X = [X_src',X_tar'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Y_src));

    %% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);
	%%% M0
	M0 = e * e' * C;  %multiply C for better normalization

    acc_ite = [];
	Y_tar_pseudo = [];
    
	if strcmp(mode,'W-BDA')
        knn_model = fitcknn(X_src,Y_src,'NumNeighbors',1);
        Y_tar_pseudo = knn_model.predict(X_tar);
    end
    %% Iteration
    for i = 1 : T
        %%% Mc
        N = 0;
        if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(n,1);
                if strcmp(mode,'W-BDA')
                    Ps = length(find(Y_src==c)) / length(Y_src);
                    Pt = length(find(Y_tar_pseudo == c)) / length(Y_tar_pseudo);
                    alpha = Pt / Ps;
                    mu = 1;
                else
                    alpha = 1;
                end
                e(Y_src==c) = 1 / length(find(Y_src==c));
                e(ns+find(Y_tar_pseudo==c)) = -alpha / length(find(Y_tar_pseudo==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end
        M = (1 - mu) * M0 + mu * N;
        M = M / norm(M,'fro');
        
        %% Calculation
        if strcmp(kernel_type,'primal')
            [A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
            Z = A'*X;
        else
            K = kernel_bda(kernel_type,X,[],gamma);
            [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
            Z = A'*K;
        end
   
        %normalization for better classification performance
		Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(X_src,1))';
        Zt = Z(:,size(X_src,1)+1:end)';

        knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
        Y_tar_pseudo = knn_model.predict(Zt);
        acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar); 
        fprintf('Iteration [%2d]:BDA+NN=%0.4f\n',i,acc);
        acc_ite = [acc_ite;acc];
	end

end

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013

function K = kernel_bda(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end