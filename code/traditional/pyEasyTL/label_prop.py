import numpy as np
from scipy.optimize import linprog
import pulp

def label_prop(C, nt, Dct, lp="linear"):
    
#Inputs:
#  C      :    Number of share classes between src and tar
#  nt     :    Number of target domain samples
#  Dct    :    All d_ct in matrix form, nt * C
#  lp     :    Type of linear programming: linear (default) | binary
#Outputs:
#  Mcj    :    all M_ct in matrix form, m * C

    intcon = C * nt
    Aeq = np.zeros([nt, intcon])
    Beq = np.ones([nt, 1])
    for i in range(nt):
        Aeq[i, i*C:(i+1)*C] = 1;
	
    D_vec = np.reshape(Dct, (1, intcon))
    CC = np.asarray(D_vec, dtype=np.double)
	
    A = np.array([])
    B = -1 * np.ones([C, 1])
    for i in range(C):
        all_zeros = np.zeros([1, intcon])
        for j in range(i, C * nt, C):
            all_zeros[0][j] = -1
        if i == 0:
            A = all_zeros
        else:
            A = np.vstack((A, all_zeros))
    
    if lp == "binary":
        print("not implemented yet!")
    else:
        res = linprog(CC,A,B,Aeq,Beq, bounds=tuple((0, 1) for _ in range(intcon)))
    Mct_vec = res.get("x")[0:C*nt]
    Mcj = Mct_vec.reshape((C,nt), order="F").T
  
    return Mcj
	