#coding=utf-8
from SOT import SOT,load_from_file
import numpy as np

if __name__=='__main__':
    tsot=SOT('ACT','./clustertemp/',19,0.5,1,3)
    Sx,Sy,Tx,Ty=load_from_file('./data/','MDA_JCPOT_ACT.json','D','H')
    spath='./data/test_MDA_JCPOT_ACT_diag_SG.json'
    tpath='./clustertemp/data/test_MDA_JCPOT_ACT_19_diag_TG.json'
    tmodelpath='./clustertemp/model/test_MDA_JCPOT_ACT_19_diag_H'
    pred,acc=tsot.fit_predict(Sx,Sy,Tx,Ty,spath,'D',tpath,tmodelpath,'H')
    print(acc)