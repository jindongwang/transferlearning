#!/usr/bin/env python3.6
''' CNN Architecture

Based on the architecture in MDD <https://github.com/thuml/MDD>,
and leverage the repositories of `domainbed` <https://github.com/facebookresearch/DomainBed>
and `pytorch_GAN_zoo` <https://github.com/facebookresearch/pytorch_GAN_zoo>.
Architectures organized and enhanced for the use for the Causal Semantic Generative model.
'''
__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

import sys, os
from warnings import warn
import torch as tc
import torch.nn as nn
import torchvision as tv
from . import backbone
from . import mlp
sys.path.append('..')
from distr import tensorify, is_same_tensor, wrap4_multi_batchdims
dbpath = '../a-domainbed/DomainBed'
if os.path.isdir(dbpath):
    sys.path.append(dbpath)
    import domainbed # 54c2f8c
    from domainbed.networks import Featurizer

def init_linear(nnseq, wmean, wstd, bval):
    for mod in nnseq:
        if type(mod) is nn.Linear:
            mod.weight.data.normal_(wmean, wstd)
            mod.bias.data.fill_(bval)

# The inference/discriminative models / encoders.
class CNNsvy1x(nn.Module):
    def __init__(self, backbone_stru: str, dim_bottleneck: int, dim_s: int, dim_y: int, dim_v: int,
            std_v1x_val: float, std_s1vx_val: float, # if <= 0, then learn the std.
            dims_bb2bn: list=None, dims_bn2s: list=None, dims_s2y: list=None,
            vbranch: bool=False, dims_bn2v: list=None):
        """ Based on MDD from <https://github.com/thuml/MDD>
        if not vbranch:
              (bb)   (bn)        (med)   (cls)
                       /->   v   -\
            x ====>  -|            |-> s ----> y
                       \-> parav -/
        else:
              (bb)   (bn)        (med)   (cls)
            x ====>  ---->       ----> s ----> y
                               \ (vbr)
                                \----> v
        """
        if not vbranch: assert dim_v <= dim_bottleneck
        super(CNNsvy1x, self).__init__()
        self.dim_s = dim_s; self.dim_v = dim_v; self.dim_y = dim_y
        self.shape_s = (dim_s,); self.shape_v = (dim_v,)
        self.vbranch = vbranch
        self.std_v1x_val = std_v1x_val; self.std_s1vx_val = std_s1vx_val
        self.learn_std_v1x = std_v1x_val <= 0 if type(std_v1x_val) is float else (std_v1x_val <= 0).any()
        self.learn_std_s1vx = std_s1vx_val <= 0 if type(std_s1vx_val) is float else (std_s1vx_val <= 0).any()

        self._x_cache_bb = self._bb_cache = None
        self._x_cache_bn = self._bn_cache = None
        self._param_groups = []

        if 'domainbed' in globals() and backbone_stru.startswith("DB"):
            self.nn_backbone = Featurizer((3,224,224),
                    {'resnet18': backbone_stru[2:]=='resnet18', 'resnet_dropout': 0.})
            self._param_groups += [{"params": self.nn_backbone.parameters(), "lr_ratio": 1.0}]
            self.nn_backbone.output_num = lambda: self.nn_backbone.n_outputs
            if dim_bottleneck is None: dim_bottleneck = self.nn_backbone.output_num() // 2
            if dim_s is None: dim_s = self.nn_backbone.output_num() // 4
        else:
            self.nn_backbone = backbone.network_dict[backbone_stru]()
            self._param_groups += [{"params": self.nn_backbone.parameters(), "lr_ratio": 0.1}]
        self.f_backbone = wrap4_multi_batchdims(self.nn_backbone, ndim_vars=3)

        if dims_bb2bn is None: dims_bb2bn = []
        self.nn_bottleneck = mlp.mlp_constructor(
                [self.nn_backbone.output_num()] + dims_bb2bn + [dim_bottleneck],
                nn.ReLU, lastactv = False
            )
        init_linear(self.nn_bottleneck, 0., 5e-3, 0.1)
        self._param_groups += [{"params": self.nn_bottleneck.parameters(), "lr_ratio": 1.}]
        self.f_bottleneck = self.nn_bottleneck

        if dims_bn2s is None: dims_bn2s = []
        self.nn_mediate = nn.Sequential(
                *([] if backbone_stru.startswith("DB") else [nn.BatchNorm1d(dim_bottleneck)]),
                nn.ReLU(),
                # nn.Dropout(0.5),
                mlp.mlp_constructor(
                    [dim_bottleneck] + dims_bn2s + [dim_s],
                    nn.ReLU, lastactv = False)
            )
        init_linear(self.nn_mediate, 0., 1e-2, 0.)
        self._param_groups += [{"params": self.nn_mediate.parameters(), "lr_ratio": 1.}]
        self.f_mediate = wrap4_multi_batchdims(self.nn_mediate, ndim_vars=1) # required by `BatchNorm1d`

        if dims_s2y is None: dims_s2y = []
        self.nn_classifier = nn.Sequential(
                nn.ReLU(),
                # nn.Dropout(0.5),
                mlp.mlp_constructor(
                    [dim_s] + dims_s2y + [dim_y],
                    nn.ReLU, lastactv = False)
            )
        init_linear(self.nn_classifier, 0., 1e-2, 0.)
        self._param_groups += [{"params": self.nn_classifier.parameters(), "lr_ratio": 1.}]
        self.f_classifier = self.nn_classifier

        if vbranch:
            if dims_bn2v is None: dims_bn2v = []
            self.nn_vbranch = nn.Sequential(
                    nn.BatchNorm1d(dim_bottleneck),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp.mlp_constructor(
                        [dim_bottleneck] + dims_bn2v + [dim_v],
                        nn.ReLU, lastactv = False)
                )
            init_linear(self.nn_vbranch, 0., 1e-2, 0.)
            self._param_groups += [{"params": self.nn_vbranch.parameters(), "lr_ratio": 1.}]
            self.f_vbranch = wrap4_multi_batchdims(self.nn_vbranch, ndim_vars=1)

        ## std models
        if self.learn_std_v1x:
            if not vbranch:
                self.nn_std_v = nn.Sequential(
                        mlp.mlp_constructor(
                            [self.nn_backbone.output_num()] + dims_bb2bn + [dim_v],
                            nn.ReLU, lastactv = False),
                        nn.Softplus()
                    )
            else:
                self.nn_std_v = nn.Sequential(
                        mlp.mlp_constructor(
                            [dim_bottleneck] + dims_bn2v + [dim_v],
                            nn.ReLU, lastactv = False),
                        nn.Softplus()
                    )
            init_linear(self.nn_std_v, 0., 1e-2, 0.)
            self._param_groups += [{"params": self.nn_std_v.parameters(), "lr_ratio": 1.}]
            self.f_std_v = self.nn_std_v

        if self.learn_std_s1vx:
            self.nn_std_s = nn.Sequential(
                    nn.BatchNorm1d(dim_bottleneck),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp.mlp_constructor(
                        [dim_bottleneck] + dims_bn2s + [dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self._param_groups += [{"params": self.nn_std_s.parameters(), "lr_ratio": 1.}]
            self.f_std_s = wrap4_multi_batchdims(self.nn_std_s, ndim_vars=1)

    def _get_bb(self, x):
        if not is_same_tensor(x, self._x_cache_bb):
            self._x_cache_bb = x
            self._bb_cache = self.f_backbone(x)
        return self._bb_cache

    def _get_bn(self, x):
        if not is_same_tensor(x, self._x_cache_bn):
            self._x_cache_bn = x
            self._bn_cache = self.f_bottleneck(self._get_bb(x))
        return self._bn_cache

    def v1x(self, x):
        bn = self._get_bn(x)
        if not self.vbranch: return bn[..., :self.dim_v]
        else: return self.f_vbranch(bn)
    def std_v1x(self, x):
        if self.learn_std_v1x:
            if not self.vbranch: return self.f_std_v(self._get_bb(x))
            else: return self.f_std_v(self._get_bn(x))
        else:
            return tensorify(x.device, self.std_v1x_val)[0].expand(x.shape[:-3]+(self.dim_v,))

    def s1vx(self, v, x):
        if not self.vbranch:
            bn = self._get_bn(x)
            bn_synth = tc.cat([v, bn[..., self.dim_v:]], dim=-1)
            return self.f_mediate(bn_synth)
        else:
            return self.s1x(x)
    def std_s1vx(self, v, x):
        if self.learn_std_s1vx:
            if not self.vbranch:
                bn = self._get_bn(x)
                bn_synth = tc.cat([v, bn[..., self.dim_v:]], dim=-1)
                return self.f_std_s(bn_synth)
            else:
                return self.std_s1x(x)
        else:
            return tensorify(x.device, self.std_s1vx_val)[0].expand(x.shape[:-3]+(self.dim_s,))

    def s1x(self, x):
        return self.f_mediate(self._get_bn(x))
    def std_s1x(self, x):
        if self.learn_std_s1vx:
            return self.f_std_s(self._get_bn(x))
        else:
            return tensorify(x.device, self.std_s1vx_val)[0].expand(x.shape[:-3]+(self.dim_s,))

    def y1s(self, s):
        return self.f_classifier(s).squeeze(-1) # squeeze for binary y

    def ys1x(self, x):
        s = self.s1x(x)
        return self.y1s(s), s

    def forward(self, x):
        return self.y1s(self.s1x(x))

    def parameter_groups(self):
        return self._param_groups

    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()

# The generative models / decoders.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

## Instances
class CNN_DCGANvar_224(nn.Module):
    # Based on the decoder of DCGAN.
    def __init__(self, dim_in, dim_feat, dim_chanl = 3):
        super(CNN_DCGANvar_224, self).__init__()
        self.nn_main = nn.Sequential(
                # l_out = stride*(l_in - 1) + l_kernel - 2*padding. (*, *, l_kernel, stride, padding)
                # input is Z, going into a convolution
                nn.ConvTranspose2d( dim_in, dim_feat * 8, 7, 1, 0, bias=False),
                nn.BatchNorm2d(dim_feat * 8),
                nn.ReLU(True),
                # state size. (dim_feat*8) x 7 x 7
                nn.ConvTranspose2d(dim_feat * 8, dim_feat * 4, 4, 4, 0, bias=False),
                nn.BatchNorm2d(dim_feat * 4),
                nn.ReLU(True),
                # state size. (dim_feat*4) x 28 x 28
                nn.ConvTranspose2d( dim_feat * 4, dim_feat * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_feat * 2),
                nn.ReLU(True),
                # state size. (dim_feat*2) x 56 x 56
                nn.ConvTranspose2d( dim_feat * 2, dim_feat, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_feat),
                nn.ReLU(True),
                # state size. (dim_feat) x 112 x 112
                nn.ConvTranspose2d( dim_feat, dim_chanl, 4, 2, 1, bias=True), # False),
                # nn.Tanh()
                # state size. (dim_chanl) x 224 x 224
            )
        self.apply(weights_init)
        self._param_groups = [{"params": self.nn_main.parameters(), "lr_ratio": 1.}]
        self.f_main = wrap4_multi_batchdims(self.nn_main, ndim_vars=3)

    def forward(self, val):
        # `val` should be of shape (..., dim_in)
        return self.f_main(val[..., None, None])

class CNN_DCGANpretr_224(nn.Module):
    def __init__(self, dim_in, dim_feat = None, dim_chanl = None):
        # if dim_feat is not None or dim_chanl is not None:
        #     warn(f"`dim_feat` {dim_feat} and `dim_chanl` {dim_chanl} ignored")
        super(CNN_DCGANpretr_224, self).__init__()
        self._param_groups = []
        self.nn_pre = nn.Sequential(
                nn.Linear(dim_in, dim_feat), nn.Tanh(),
                nn.Linear(dim_feat, 120), nn.Tanh()
            )
        self.nn_pre.apply(weights_init)
        self._param_groups += [{"params": self.nn_pre.parameters(), "lr_ratio": 1.}]
        self.f_pre = self.nn_pre

        self.nn_backbone = tc.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', # force_reload=True,
                pretrained=True, useGPU=False, model_name='cifar10').getOriginalG()
        self._param_groups += [{"params": self.nn_backbone.parameters(), "lr_ratio": 0.1}]
        self.f_backbone = wrap4_multi_batchdims(self.nn_backbone, ndim_vars=1)

        self.nn_post = nn.Sequential(
                nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(3, 3, 4, 4, 16, bias=False)
            )
        self.nn_post.apply(weights_init)
        self._param_groups += [{"params": self.nn_post.parameters(), "lr_ratio": 1.}]
        self.f_post = wrap4_multi_batchdims(self.nn_post, ndim_vars=3)

    def forward(self, val):
        # `val` should be of shape (..., dim_in)
        return self.f_post(self.f_backbone(self.f_pre(val)))

    def parameter_groups(self):
        return self._param_groups

class CNN_PGANpretr_224(nn.Module):
    def __init__(self, dim_in, dim_feat = None, dim_chanl = None):
        # if dim_feat is not None or dim_chanl is not None:
        #     warn(f"`dim_feat` {dim_feat} and `dim_chanl` {dim_chanl} ignored")
        super(CNN_PGANpretr_224, self).__init__()
        self._param_groups = []
        self.nn_pre = nn.Sequential(
                nn.Linear(dim_in, dim_feat), nn.Tanh(),
                nn.Linear(dim_feat, 512), nn.Tanh()
            )
        self.nn_pre.apply(weights_init)
        self._param_groups += [{"params": self.nn_pre.parameters(), "lr_ratio": 1.}]
        self.f_pre = self.nn_pre

        self.nn_backbone = tc.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', # force_reload=True,
                pretrained=True, useGPU=False, model_name='celebAHQ-256').getOriginalG() # 'cifar10' unavailable for PGAN
        self._param_groups += [{"params": self.nn_backbone.parameters(), "lr_ratio": 0.1}]
        self.f_backbone = wrap4_multi_batchdims(self.nn_backbone, ndim_vars=1)

        self.f_post = tv.transforms.CenterCrop(224)

    def forward(self, val):
        # `val` should be of shape (..., dim_in)
        return self.f_post(self.f_backbone(self.f_pre(val)))

    def parameter_groups(self):
        return self._param_groups

## Uniform interfaces
class CNNx1sv(nn.Module):
    def __init__(self, dim_xside: int, dim_s: int, dim_v: int, dim_feat: int, dectype: str="DCGANvar"):
        super(CNNx1sv, self).__init__()
        self.net = globals()["CNN_" + dectype + "_" + str(dim_xside)](dim_s + dim_v, dim_feat)
        # Parameters of `self.net` automatically included in `self.parameters()`

    def x1sv(self, s, v):
        return self.net(tc.cat([s, v], dim=-1))
    def forward(self, s, v): return self.x1sv(s, v)
    def parameter_groups(self):
        return self.net._param_groups

    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()

class CNNx1s(nn.Module):
    def __init__(self, dim_xside: int, dim_s: int, dim_feat: int, dectype: str="DCGANvar"):
        super(CNNx1s, self).__init__()
        self.net = globals()["CNN_" + dectype + "_" + str(dim_xside)](dim_s, dim_feat)
        # Parameters of `self.net` automatically included in `self.parameters()`

    def x1s(self, s): return self.net(s)
    def forward(self, s): return self.x1s(s)
    def parameter_groups(self):
        return self.net._param_groups

    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()

