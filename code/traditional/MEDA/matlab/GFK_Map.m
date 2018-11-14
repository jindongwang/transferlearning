function [Xs_new,Xt_new,G] = GFK_Map(Xs,Xt,dim)
    Ps = pca(Xs);
    Pt = pca(Xt);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    sq_G = real(G^(0.5));
    Xs_new = (sq_G * Xs')';
    Xt_new = (sq_G * Xt')';
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