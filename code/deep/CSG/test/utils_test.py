#!/usr/bin/env python3.6
import sys
from time import time
import numpy as np
sys.path.append('..')
import utils
'''Test cases of the 'distr' package.
'''

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

shape = (500, 200, 50)

arr = np.random.rand(*shape)
length = arr.shape[-1]
for n_win in range(1, 3*shape[-1] + 2):
    print(f"{n_win:3d}", end=", ")
    lwid = (n_win - 1) // 2
    rwid = n_win//2 + 1
    t = time(); ma_slim = utils.moving_average_slim(arr, n_win); print(f"{time() - t:.6f}", end=", ")
    t = time(); ma_full = utils.moving_average_full(arr, n_win); print(f"{time() - t:.6f}", end=", ")
    t = time(); ma_full_check = utils.moving_average_full_checker(arr, n_win); print(f"{time() - t:.6f}", end=", ")
    ma_slim_check = ma_full_check[..., lwid: length-rwid+1]
    # print(ma_slim.shape, ma_slim_check.shape, ma_full.shape, ma_full_check.shape)
    print(f"slim: {np.allclose(ma_slim, ma_slim_check)}, full: {np.allclose(ma_full, ma_full_check)}")

