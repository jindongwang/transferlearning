import numpy as np
import pulp

def label_prop(C, nt, Dct, lp="linear"):
    
#Inputs:
#  C      :    Number of share classes between src and tar
#  nt     :    Number of target domain samples
#  Dct    :    All d_ct in matrix form, nt * C
#  lp     :    Type of linear programming: linear (default) | binary
#Outputs:
#  Mcj    :    all M_ct in matrix form, m * C
    
    Dct = abs(Dct)
    model = pulp.LpProblem("Cost minimising problem", pulp.LpMinimize)
    Mcj = pulp.LpVariable.dicts("Probability",
                                ((i, j) for i in range(C) for j in range(nt)),
                                lowBound=0,
                                upBound=1,
                                cat='Continuous')
    
    # Objective Function
    model += (
    pulp.lpSum([Dct[j, i]*Mcj[(i, j)] for i in range(C) for j in range(nt)])
    )
    
    # Constraints
    for j in range(nt):
        model += pulp.lpSum([Mcj[(i, j)] for i in range(C)]) == 1
    for i in range(C):
        model += pulp.lpSum([Mcj[(i, j)] for j in range(nt)]) >= 1
    
    # Solve our problem
    model.solve()
    pulp.LpStatus[model.status]
    Output = [[Mcj[i, j].varValue for i in range(C)] for j in range(nt)]
    
    return np.array(Output)
	