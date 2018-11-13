function [acc,G,Cls] = GFK(X_src,Y_src,X_tar,Y_tar,dim)
% This is the implementation of Geodesic Flow Kernel.
% Reference: Boqing Gong et al. Geodesic flow kernel for Unsupervised Domain Adaptation. CVPR 2012.
   
% Inputs:
%%% X_src  :   source feature matrix, ns * n_feature
%%% Y_src  :   source label vector, ns * 1
%%% X_tar  :   target feature matrix, nt * n_feature
%%% Y_tar  :   target label vector, nt * 1
%%% dim    :   dimension of geodesic flow kernel, dim <= 0.5 * n_feature

% Outputs:
%%% acc    :   accuracy after GFK and 1NN
%%% G      :   geodesic flow kernel matrix
%%% Cls    :   prediction labels for target, nt * 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Ps = pca(X_src);
    Pt = pca(X_tar);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    [Cls, acc] = my_kernel_knn(G, X_src, Y_src, X_tar, Y_tar);
end


function [prediction,accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
    dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
        + repmat(diag(Xt*M*Xt')',length(Yr),1)...
        - 2*Xr*M*Xt';
    [~, minIDX] = min(dist);
    prediction = Yr(minIDX);
    accuracy = sum( prediction==Yt ) / length(Yt); 
end

function G = GFK_core(Q,Pt)
    % Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
    %        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
    % Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

    % ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
    % B. Gong, Y. Shi, F. Sha, and K. Grauman.  
    % Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

    % Contact: Boqing Gong (boqinggo@usc.edu)

    N = size(Q,2); % 
    dim = size(Pt,2);

    % compute the principal angles
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    V2 = -V2;
    theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

    % compute the geodesic flow kernel
    eps = 1e-20;
    B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
    B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
    B3 = B2;
    B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
    G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
        * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
        * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';
end