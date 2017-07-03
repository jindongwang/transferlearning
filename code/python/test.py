#encoding=utf-8
"""
    Created on 17:22 2017/5/1 
    @author: Jindong Wang
"""

import da_tool.tca
import numpy as np

file_path = 'data/test_tca_data.csv'
data = np.loadtxt(file_path, delimiter=',')
x_src = data[:, :81]
x_tar = data[:, 81:]

# example usage of TCA
my_tca = da_tool.tca.TCA(dim=30,kerneltype='rbf', kernelparam=1, mu=1)
x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(x_src, x_tar)
np.savetxt('x_src1.csv', x_src_tca, delimiter=',', fmt='%.6f')
np.savetxt('x_tar1.csv', x_tar_tca, delimiter=',', fmt='%.6f')