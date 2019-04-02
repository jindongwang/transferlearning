function [Xs_new,Xt_new] = GFK_map(Xs,Xt)
% Inputs:
%%% X_src  :source feature matrix, ns * m
%%% Y_src  :source label vector, ns * 1
%%% X_tar  :target feature matrix, nt * m
%%% Y_tar  :target label vector, nt * 1
% Outputs:
%%% acc    :accuracy after GFK and 1NN
%%% G      :geodesic flow kernel matrix

    Ps = pca(Xs);
    Pt = pca(Xt);
    dim = getGFKDim(Xs,Xt);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    sq_G = real(G^(0.5));
    Xs_new = (sq_G * Xs')';
    Xt_new = (sq_G * Xt')';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% Copyright (c) 2013, Basura Fernando
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, 
% are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this 
%list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation and/or 
% other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% This is the subspace dimensionality estimation method proposed in
% B. Gong, Y. Shi, F. Sha, and K. Grauman. Geodesic flow kernel for unsupervised domain adaptation. In CVPR, 2012.
% 
% Please cite
% 
% @inproceedings{Fernando2013b,
% author = {Basura Fernando, Amaury Habrard, Marc Sebban, Tinne Tuytelaars},
% title = {Unsupervised Visual Domain Adaptation Using Subspace Alignment},
% booktitle = {ICCV},
% year = {2013},
% } 
%
function RES = getGFKDim(Source_Data,Target_Data,Pss,Pts,Psstt)

if nargin < 3
    [Pss,D1,E1] = pca(Source_Data);
end
if nargin < 4
    [Pts,D2,E2] =  pca(Target_Data);
end
if nargin < 5
    [Psstt,D2,E2] = pca([Source_Data;Target_Data]);
end

    DIM = round(size(Source_Data,2) * 0.5) ;  % if the dimensionality is large set DIM to say 200    
    RES = -1;
    for d = 1 : DIM
        Ps = Pss(:,1:d);
        Pt = Pts(:,1:d);
        Pst = Psstt(:,1:d);
        alpha1 = getAngle(Ps,Pst,d);
        alpha2 = getAngle(Pt,Pst,d);
        D = (alpha1 + alpha2)*0.5;
        for dd = 1 : d
            if(round(D(1,dd)*100) == 100)                
                RES =d;
                return;
            end
        end        
    end
end

function alpha = getAngle(Ps,Pt,DD)

    Q = [Ps, null(Ps')];
    N = size(Q,2); % N is the
    dim = size(Pt,2);
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:)); 
    alpha = zeros(1,DD);
    for i = 1 : DD
        alpha(1,i) =sind(real(acosd(Gam(i,i))));
    end
end

