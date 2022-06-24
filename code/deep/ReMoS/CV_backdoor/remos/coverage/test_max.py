import numpy as np
from pdb import set_trace as st


def get_max_values(array, k):
    indexs = tuple()
    while k > 0:
        idx = np.argwhere(array == array.max())
        idx = np.ascontiguousarray(idx).reshape(-1)
        # indexs.append(idx)
        indexs += (idx,)
        
        array[idx] = -np.inf
        
        print(idx.shape)
        k -= len(idx)

    indexs = np.concatenate(indexs)
    # return indexs
    st()
        
        
a = np.array([5,12,32,659,-4,5,23,659]).astype(np.float32)

np.sort(a)
st()
get_max_values(a, 4)