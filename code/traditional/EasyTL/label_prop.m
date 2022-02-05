function [Mcj] = label_prop(C,nt,Dct,lp)
% Inputs:
%%% C      :    Number of share classes between src and tar
%%% nt     :    Number of target domain samples
%%% Dct    :    All d_ct in matrix form, nt * C
%%% lp     :    Type of linear programming: linear (default) | binary
% Outputs:
%%% Mcj    :    all M_ct in matrix form, m * C

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    intcon = C * nt;      % Number of all variables to be solved (all x_ct and o_t)
    if nargin == 3
        lp = 'linear';
    end
   
    Aeq = zeros(nt, intcon);
    
    Beq = ones(nt ,1);
    for i = 1 : nt 
        Aeq(i,(i - 1) * C + 1 : i * C) = 1;
    end
    %for i = 1 : nt 
   %     lllll =i
   %     all_zeros = zeros(1,intcon);
   %     all_zeros((i - 1) * C + 1 : i * C) = 1;
   %     Aeq = [Aeq;all_zeros];
   %     Beq = [Beq;1];
   % end
    D_vec = reshape(Dct',1,intcon);
    CC = double(D_vec);
    
   
   
    
   
    A = [];
    B = [];
    for i = 1 : C
        all_zeros = zeros(1,intcon);
        j = i : C : C * nt;
        all_zeros(j) = -1;
        A = [A;all_zeros];
        B = [B;-1];
    end

    lb_12 = zeros(intcon,1);
    ub_12 = ones(intcon,1);
    
  
%     options = optimoptions('linprog','Algorithm','interior-point');

    if strcmp(lp,'binary')
       
        X = intlinprog(CC,intcon,A,B,Aeq,Beq,lb_12,ub_12);
    else
        
        X = linprog(CC,A,B,Aeq,Beq,lb_12,ub_12);
    end
    Mct_vec = X(1:C*nt);
    Mcj = reshape(Mct_vec,C,nt)';   % M matrix, nt * C
end