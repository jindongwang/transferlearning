#!/usr/bin/env python3.6
'''Probabilistic Programming Package.

The prototype is distributions, which can be a conditional one with
functions for parameters to define the dependency. Distribution
multiplication is implemented, as well as the mean, expectation,
sampling with backprop capability, and log-probability.
'''

__author__ = "Chang Liu"
__version__ = "1.0.1"
__email__ = "changliu@microsoft.com"

from .base import Distr, DistrElem
from .instances import Determ, Normal, MVNormal, Catg, Bern

from .utils import ( append_attrs,
        edic, edicify,
        fargnames, fedic, wrap4_multi_batchdims,
        tensorify, is_scalar, is_same_tensor,
        expand_front, flatten_last, reduce_last, swap_dim_ranges, expand_middle,
        tcsizeify, tcsize_div, tcsize_broadcast,
    )
from .tools import elbo, elbo_z2xy, elbo_z2xy_twist, elbo_zy2x

