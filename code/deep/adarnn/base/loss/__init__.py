from base.loss.adv_loss import adv
from base.loss.coral import CORAL
from base.loss.cos import cosine
from base.loss.kl_js import kl_div, js
from base.loss.mmd import MMD_loss
from base.loss.mutual_info import Mine
from base.loss.pair_dist import pairwise_dist

__all__ = [
    'adv',
    'CORAL',
    'cosine',
    'kl_div',
    'js'
    'MMD_loss',
    'Mine',
    'pairwise_dist'
]