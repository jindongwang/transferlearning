#!/usr/bin/env python3.6
import sys
import torch as tc
sys.path.append('..')
import distr as ds
from distr.utils import expand_front, swap_dim_ranges
'''Test cases of the 'distr' package.
'''

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

shape_x = (1,2)
shape_bat = (3,4)
device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
ds.Distr.default_device = device

def test_fun(title, p_z, p_x1z):
    print(title)
    print("p_z:", p_z.names, p_z.parents)
    print("p_x1z:", p_x1z.names, p_x1z.parents)
    p_zx = p_z * p_x1z
    print("p_zx:", p_zx.names, p_zx.parents)
    smp_z = p_z.draw(shape_bat)
    print("sample shape z:", smp_z['z'].shape)
    smp_x1z = p_x1z.draw((), smp_z)
    print("sample shape x:", smp_x1z['x'].shape)
    print("logp match:", tc.allclose(
        p_z.logp(smp_z) + p_x1z.logp(smp_x1z, smp_z),
        p_zx.logp(smp_z|smp_x1z) ))
    smp_zx = p_zx.draw(shape_bat)
    print("sample shape z:", smp_zx['z'].shape)
    print("sample shape x:", smp_zx['x'].shape)
    print("logp match:", tc.allclose(
        p_z.logp(smp_zx) + p_x1z.logp(smp_zx, smp_zx),
        p_zx.logp(smp_zx) ))
    print("logp_cartes shape:", p_x1z.logp_cartes(smp_x1z, smp_z).shape)
    print()
    ds.Distr.clear()

# Normal
ndim_x = len(shape_x)
test_fun("Normal:",
        p_z = ds.Normal('z', 0., 1.),
        p_x1z = ds.Normal('x', shape = shape_x, mean =
            lambda z: swap_dim_ranges( expand_front(z, shape_x), (0, ndim_x), (ndim_x, ndim_x+z.ndim) ),
            std = 1.
        ))

# MVNormal
test_fun("MVNormal:",
        p_z = ds.MVNormal('z', 0., 1.),
        p_x1z = ds.MVNormal('x', shape = shape_x, mean =
            lambda z: swap_dim_ranges( expand_front(z, shape_x).squeeze(-1), (0, ndim_x), (ndim_x, ndim_x+z.ndim-1) ),
            cov = 1.
        ))

# Catg
ncat_z = 3
ncat_x = 4
w_z = tc.rand(ncat_z)
w_z = w_z / w_z.sum()
w_x = tc.rand((ncat_z,) + shape_x + (ncat_x,), device=device)
w_x = w_x / w_x.sum(dim=-1, keepdim=True)
test_fun("Catg:",
        p_z = ds.Catg('z', probs = w_z),
        p_x1z = ds.Catg('x', shape = shape_x, probs =
            lambda z: w_x.index_select(dim=0, index=z.flatten()).reshape(z.shape + w_x.shape[1:])
        ))

# Bern
w_x = tc.rand(shape_x, device=device)
w_x = tc.stack([1-w_x, w_x], dim=0)
test_fun("Bern:",
        p_z = ds.Bern('z', probs = tc.rand(())),
        p_x1z = ds.Bern('x', shape = shape_x, probs =
            lambda z: w_x.index_select(dim=0, index=z.flatten()).reshape(z.shape + w_x.shape[1:])
        ))

