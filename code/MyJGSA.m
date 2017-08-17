function [acc,acc_list,A,B] = MyJGSA(X_src,Y_src,X_tar,Y_tar,options)
%% Joint Geometrical and Statistic Adaptation
% Inputs:
%%% X_src  :source feature matrix, ns * m
%%% Y_src  :source label vector, ns * 1
%%% X_tar  :target feature matrix, nt * m
%%% Y_tar  :target label vector, nt * 1
%%% options:option struct
% Outputs:
%%% acc    :final accuracy using knn, float
%%% acc_list:list of all accuracies during iterations
%%% A      :final adaptation matrix for source domain, m * dim
%%% B      :final adaptation matrix for target domain, m * dim

    alpha = options.alpha;
	mu = options.mu;
	beta = options.beta;
	gamma = options.gamma;
	kernel_type = options.kernel_type;
	dim = options.dim;
	T = options.T;

    X_src = X_src';
    X_tar = X_tar';
    
	m = size(X_src,1);
	ns = size(X_src,2);
	nt = size(X_tar,2);

	class_set = unique(Y_src);
	C = length(class_set);
    acc_list = [];
    Y_tar_pseudo = [];
	if strcmp(kernel_type,'primal')
		[Sw, Sb] = scatter_matrix(X_src',Y_src);
		P = zeros(2 * m,2 * m);
		P(1:m,1:m) = Sb;
		Q = Sw;

		for t = 1 : T
			[Ms,Mt,Mst,Mts] = construct_mmd(ns,nt,Y_src,Y_tar_pseudo,C);
			Ts = X_src * Ms * X_src';
	        Tt = X_tar * Mt * X_tar';
	        Tst = X_src * Mst * X_tar';
	        Tts = X_tar * Mts * X_src';

	        Ht = eye(nt) - 1 / nt * ones(nt,nt);

	        X = [zeros(m,ns),zeros(m,nt);zeros(m,ns),X_tar];
	        H = [zeros(ns,ns),zeros(ns,nt);zeros(nt,ns),Ht];

	        Smax = mu * X * H * X' + beta * P;
	        Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
                Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
            mm = 1e-9*eye(2*m);
	        [W,~] = eigs(Smax,Smin + mm,dim,'LM');
	        As = W(1:m,:);
	        At = W(m+1:end,:);
	        Zs = (As' * X_src)';
	        Zt = (At' * X_tar)';
	        
	        if T > 1
	            knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
		        Y_tar_pseudo = knn_model.predict(Zt);
		        acc = length(find(Y_tar_pseudo == Y_tar)) / length(Y_tar);
	            fprintf('acc of iter %d: %0.4f\n',t, acc);
                acc_list = [acc_list;acc];
	        end
		end
	else
		Xst = [X_src,X_tar];
		nst = size(Xst,2);
		[Ks, Kt, Kst] = constructKernel(X_src,X_tar,kernel_type,gamma);
		[Sw, Sb] = scatter_matrix(Ks,Y_src);
		P = zeros(2 * nst,2 * nst);
		P(1:nst,1:nst) = Sb;
		Q = Sw;
		for t = 1:T

	        % Construct MMD matrix
	        [Ms, Mt, Mst, Mts] = construct_mmd(ns,nt,Y_src,Y_tar_pseudo,C);
	        
	        Ts = Ks'*Ms*Ks;
	        Tt = Kt'*Mt*Kt;
	        Tst = Ks'*Mst*Kt;
	        Tts = Kt'*Mts*Ks;

	        K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
	        Smax =  mu*K'*K+beta*P;
	        
	        Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
	                Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
	        [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), dim, 'LM');
	        W = real(W);

	        As = W(1:nst, :);
	        At = W(nst+1:end, :);

	        Zs = (As'*Ks')';
	        Zt = (At'*Kt')';

	        if T > 1
	            knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
		        Y_tar_pseudo = knn_model.predict(Zt);
		        acc = length(find(Y_tar_pseudo == Y_tar)) / length(Y_tar);
	            fprintf('acc of iter %d: %0.4f\n',t, full(acc));
                acc_list = [acc_list;acc];
	        end
	    end
    end
    A = As;
    B = At;
end

function [Sw,Sb] = scatter_matrix(X,Y)
%% Within and between class Scatter matrix
%% Inputs:
%%% X: data matrix, length * dim
%%% Y: label vector, length * 1
% Outputs:
%%% Sw: With-in class matrix, dim * dim
%%% Sb: Between class matrix, dim * dim
    X = X';
	dim = size(X,1);
	class_set = unique(Y);
	C = length(class_set);
	mean_total = mean(X,2);
	Sw = zeros(dim,dim);
	Sb = zeros(dim,dim);
	for i = 1 : C
		Xi = X(:,Y == class_set(i));
		mean_class_i = mean(Xi,2);
		Hi = eye(size(Xi,2)) - 1/(size(Xi,2)) * ones(size(Xi,2),size(Xi,2));
		Sw = Sw + Xi * Hi * Xi';
		Sb = Sb + size(Xi,2) * (mean_class_i - mean_total) * (mean_class_i - mean_total)';
	end
end

function [Ms,Mt,Mst,Mts] = construct_mmd(ns,nt,Y_src,Y_tar_pseudo,C)
	es = 1 / ns * ones(ns,1);
	et = -1 / nt * ones(nt,1);
	e = [es;et];

	M = e * e' * C;
	Ms = es * es' * C;
	Mt = et * et' * C;
	Mst = es * et' * C;
	Mts = et * es' * C;

	if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo) == nt
		for c = reshape(unique(Y_src),1,C)
			es = zeros(ns,1);
			et = zeros(nt,1);
			es(Y_src == c) = 1 / length(find(Y_src == c));
			et(Y_tar_pseudo == c) = -1 / length(find(Y_tar_pseudo == c));
			es(isinf(es)) = 0;
			et(isinf(et)) = 0;
			Ms = Ms + es * es';
			Mt = Mt + et * et';
			Mst = Mst + es * et';
			Mts = Mts + et * es';
		end
	end

	Ms = Ms / norm(M,'fro');
	Mt = Mt / norm(M,'fro');
	Mst = Mst / norm(M,'fro');
	Mts = Mts / norm(M,'fro');
end

function [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma)
	Xst = [Xs, Xt];   
	ns = size(Xs,2);
	nt = size(Xt,2);
	nst = size(Xst,2); 
	Kst0 = km_kernel(Xst',Xst',ker,gamma);
	Ks0 = km_kernel(Xs',Xst',ker,gamma);
	Kt0 = km_kernel(Xt',Xst',ker,gamma);

	oneNst = ones(nst,nst)/nst;
	oneN=ones(ns,nst)/nst;
	oneMtrN=ones(nt,nst)/nst;
	Ks=Ks0-oneN*Kst0-Ks0*oneNst+oneN*Kst0*oneNst;
	Kt=Kt0-oneMtrN*Kst0-Kt0*oneNst+oneMtrN*Kst0*oneNst;
	Kst=Kst0-oneNst*Kst0-Kst0*oneNst+oneNst*Kst0*oneNst;
end

function K = km_kernel(X1,X2,ktype,kpar)
% KM_KERNEL calculates the kernel matrix between two data sets.
% Input:	- X1, X2: data matrices in row format (data as rows)
%			- ktype: string representing kernel type
%			- kpar: vector containing the kernel parameters
% Output:	- K: kernel matrix
% USAGE: K = km_kernel(X1,X2,ktype,kpar)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2012.
%
% This file is part of the Kernel Methods Toolbox for MATLAB.
% https://github.com/steven2358/kmbox

	switch ktype
		case 'gauss'	% Gaussian kernel
			sgm = kpar;	% kernel width
			
			dim1 = size(X1,1);
			dim2 = size(X2,1);
			
			norms1 = sum(X1.^2,2);
			norms2 = sum(X2.^2,2);
			
			mat1 = repmat(norms1,1,dim2);
			mat2 = repmat(norms2',dim1,1);
			
			distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
	        sgm = sgm / mean(mean(distmat)); % added by jing 24/09/2016, median-distance
			K = exp(-distmat/(2*sgm^2));
			
		case 'gauss-diag'	% only diagonal of Gaussian kernel
			sgm = kpar;	% kernel width
			K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
			
		case 'poly'	% polynomial kernel
	% 		p = kpar(1);	% polynome order
	% 		c = kpar(2);	% additive constant
	        p = kpar; % jing
	        c = 1; % jing
			
			K = (X1*X2' + c).^p;
			
		case 'linear' % linear kernel
			K = X1*X2';
			
		otherwise	% default case
			error ('unknown kernel type')
	end
end