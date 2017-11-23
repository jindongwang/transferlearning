function W = feature_map(Xct,S,T,C,m)
% Get the feature transformation W as in Eq.(7)
% Inputs:
%%% Xct   :   matrix of x_ct, m * C
%%% S     :   center matrix of the source domain (all S_c s), D * C
%%% T     :   target samples matrix, D * m
%%% C     :   number of shared classes between domains
%%% m     :   number of target domain samples
% Outpus:
%%% W     :   feature mapping, D * D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Construct original Ps and Pt when Xct=1
    Ss = [];
    for i = 1 : C
        Ss = [Ss,S(:,i)];
    end
    Ps = repmat(Ss,1,m);
    Pt = [];
    for i = 1 : m
        Tc = repmat(T(:,i),1,C);
        Pt = [Pt,Tc];
    end
    
    %% Obtain the real Ps and Pt according to Xct=1 or 0
    X_vec = reshape(Xct',1,C * m);
    one_index = X_vec == 1;
    L = sum(X_vec);
    Ps = Ps(:,one_index);
    Pt = Pt(:,one_index);
    
    %% Optimize
%     f = 1 / 2 * norm(W * Ps - Pt,'fro');
%     df_dW = W * (Ps * Ps') - Pt * Ps';
    W = Pt * Ps' * pinv(Ps * Ps');     % Only set the derivative to 0
end