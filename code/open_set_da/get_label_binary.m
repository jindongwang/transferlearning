function [Xct,Ot] = get_label_binary(C,m,lambda,Dct)
% Inputs:
%%% C      :    Number of share classes between src and tar
%%% m      :    Number of target domain samples
%%% lambda :    lambda
%%% Dct    :    all d_ct in matrix form, m * C
% Outputs:
%%% Xct    :    all x_ct in matrix form, m * C
%%% Ot     :    all o_t in vector, m * 1
    intcon = C*m+m;      % Number of all variables to be solved (all x_ct and o_t)
    
    %% Construct objective
    Lambda = ones(1,m) * lambda;
    D_vec = reshape(Dct',1,C*m);
    one_vec = ones(1,m);
    CC = [D_vec,Lambda];
    
    %% Construct equalities: \sum_c x_{ct} + o_t = 1 for each t
    Aeq = [];
    Beq = [];
    for i = 1 : m
        all_zeros = zeros(1,intcon);
        all_zeros((i - 1) * C + 1 : (i - 1) * C + 10) = 1;
        all_zeros(C * m + i) = 1;
        Aeq = [Aeq;all_zeros];
        Beq = [Beq;1];
    end
    
    %% Construct inequalities: \sum_t x_{ct} \ge 1 for each c
    A = [];
    B = [];
    for i = 1 : C
        all_zeros = zeros(1,intcon);
        j = i : C : C * m;
        all_zeros(j) = -1;
        A = [A;all_zeros];
        B = [B;-1];
    end
%     A = [-1 * ones(1,C * m),zeros(1,m)];
%     B = -1 * C;
    %% Make sure x_ct and o_t are either 0 or 1
    lb_12 = zeros(intcon,1);
    ub_12 = ones(intcon,1);
    
    %% Solve integer 0/1 programming using Matlab's intlinprog function
    X = intlinprog(CC,intcon,A,B,Aeq,Beq,lb_12,ub_12);
    Xct_vec = X(1:C*m);
    Xct = reshape(Xct_vec,C,m)';   % x_ct matrix, m * C
    Ot = X(C*m+1:end);             % o_t matrix, m * 1
end