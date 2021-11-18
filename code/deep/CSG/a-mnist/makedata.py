#!/usr/bin/env python3.6
'''For generating MNIST-01 and its shifted interventional datasets.
'''
import torch as tc
import torchvision as tv
import torchvision.transforms.functional as tvtf
import argparse

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def select_xy(dataset, selected_y = (0,1), piltransf = None, ytransf = None):
    dataset_selected = [(
            tvtf.to_tensor( img if piltransf is None else piltransf(img, label) ),
            label if ytransf is None else ytransf(label)
        ) for img, label in dataset if label in selected_y]
    xs, ys = tuple(zip(*dataset_selected))
    return tc.cat(xs, dim=0), tc.tensor(ys)

def get_shift_transf(pleft: list, distr: str, loc: float, scale: float):
    return lambda img, label: tvtf.affine(img, angle=0, translate=(
            scale * getattr(tc, distr)(()) + loc * (1. - 2. * tc.bernoulli(tc.tensor(pleft[label]))), 0.
        ), scale=1., shear=0, fillcolor=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type = str, choices = {"train", "test"})
    parser.add_argument("--pleft", type = float, nargs = '+', default = [0.5, 0.5])
    parser.add_argument("--distr", type = str, choices = {"randn", "rand"})
    parser.add_argument("--loc", type = float, default = 4.)
    parser.add_argument("--scale", type = float, default = 1.)
    parser.add_argument("--procroot", type = str, default = "./data/MNIST/processed/")
    ag = parser.parse_args()

    dataset = tv.datasets.MNIST(root="./data", train = ag.mode=="train", download=True) # as PIL
    piltransf = get_shift_transf(ag.pleft, ag.distr, ag.loc, ag.scale)
    selected_y = tuple(range(len(ag.pleft)))
    shift_x, shift_y = select_xy(dataset, selected_y, piltransf)
    filename = ag.procroot + ag.mode + "".join(str(y) for y in selected_y) + (
            "_" + "_".join(f"{p:.1f}" for p in ag.pleft) +
            "_" + ag.distr + f"_{ag.loc:.1f}_{ag.scale:.1f}.pt" )
    tc.save((shift_x, shift_y), filename)
    print("Processed data saved to '" + filename + "'")

