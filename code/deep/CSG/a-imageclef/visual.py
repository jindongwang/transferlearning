#!/usr/bin/env python3.6
import warnings
import sys
sys.path.append("..")
from utils.utils_main import main_stem, get_parser, is_ood, process_continue_run, get_models, da_methods, ResultsContainer, ood_methods
from utils.preprocess import data_loader
from utils.utils import boolstr

from distr import edic
from arch import mlp, cnn
from methods import CNBBLoss, SemVar, SupVAE
from utils import Averager, unique_filename, boolstr, zip_longer # This imports from 'utils/__init__.py'

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

import torch as tc
from functools import partial
import os

MODES_TWIST = {"svgm-ind", "svae-da", "svgm-da"}

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)
from torchvision import models, transforms

def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_visual(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader, val_src_loader,
        ls_ts_tgt_loader = None, # for ood
        tr_tgt_loader = None, ts_tgt_loader = None, testdom = None # for da
    ):
    print(ag)
    IS_OOD = is_ood(ag.mode)
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Datasets
    dim_x = tc.tensor(shape_x).prod().item()
    if IS_OOD: n_per_epk = len(tr_src_loader)
    else: n_per_epk = max(len(tr_src_loader), len(tr_tgt_loader))

    # Models
    res = get_models(archtype, edic(locals()) | vars(ag), ckpt, device)
    if ag.mode.endswith("-da2"):
        discr, gen, frame, discr_src = res
        discr_src.train()
    else:
        discr, gen, frame = res

    # get pictures
    discr.eval()
    if gen is not None: gen.eval()

    # Methods and Losses
    if IS_OOD:
        lossfn = ood_methods(discr, frame, ag, dim_y, cnbb_actv="Sigmoid") # Actually the activation is ReLU, but there is no `is_treat` rule for ReLU in CNBB.
        domdisc = None
    else:
        lossfn, domdisc, dalossobj = da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt,
                discr_src if ag.mode.endswith("-da2") else None)

    epk0 = 1; i_bat0 = 1
    if ckpt is not None:
        epk0 = ckpt['epochs'][-1] + 1 if ckpt['epochs'] else 1
        i_bat0 = ckpt['i_bat']
    res = ResultsContainer( len(ag.testdoms) if IS_OOD else None,
            frame, ag, dim_y==1, device, ckpt )
    print(f"Run in mode '{ag.mode}' for {ag.n_epk:3d} epochs:")
    try:
        if ag.mode.endswith("-da2"): discr_src.eval(); true_discr = discr_src
        elif ag.mode in MODES_TWIST and ag.true_sup_val: true_discr = partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
        else: true_discr = discr
        res.evaluate(true_discr, "val "+str(ag.traindom), 'val', val_src_loader, 'src')

        if IS_OOD:
            for i, (testdom, ts_tgt_loader) in enumerate(zip(ag.testdoms, ls_ts_tgt_loader)):
                res.evaluate(discr, "test "+str(testdom), 'ts', ts_tgt_loader, 'tgt', i)
        else:
            res.evaluate(discr, "test "+str(testdom), 'ts', ts_tgt_loader, 'tgt')
            print()

        def batch_predict(images):
            import torch.nn.functional as F
            if tc.tensor(images[0]).size()[-1] == 3:
                images = [tc.tensor(pic, dtype=tc.float).permute(2, 0, 1) for pic in images]
            batch = tc.stack(tuple(i for i in images), dim=0)
            batch = batch.to(device)

            logits = discr(batch)
            probs = F.softmax(logits, dim=1)

            return probs.detach().cpu().numpy()

        if IS_OOD:
            test_loader = ls_ts_tgt_loader[0]
        else:
            test_loader = ts_tgt_loader

        iter_tr, iter_ts = iter(tr_src_loader), iter(test_loader)
        train_batch, train_label = next(iter_tr)
        test_batch, test_label = next(iter_ts)

        os.makedirs(ag.mode, exist_ok=True)

        # search for the first accurate predict:
        cursor_train, cursor_test = 0, 0
        for i in range(400):
            cursor_test += 1
            if cursor_test >= test_batch.size()[0]:
                cursor_test = 0
                test_batch, test_label = next(iter_ts)
            while True:
                x_test = test_batch[cursor_test]
                test_pred = batch_predict([x_test])
                if cursor_test < test_batch.size()[0] and test_label[cursor_test] == test_pred.squeeze().argmax():
                    break;
                else:
                    cursor_test = cursor_test + 1
                    if cursor_test >= test_batch.size()[0]:
                        cursor_test = 0
                        test_batch, test_label = next(iter_ts)

            selected_pic, selected_label = test_batch[cursor_test], test_label[cursor_test]

            cursor_train += 1
            if cursor_train >= train_batch.size()[0]:
                cursor_train = 0
                train_batch, train_label = next(iter_tr)
            while True:
                x_train = train_batch[cursor_train]
                test_pred = batch_predict([x_train])
                if cursor_train < train_batch.size()[0] and train_label[cursor_train] == test_pred.squeeze().argmax():
                    break;
                else:
                    cursor_train = cursor_train + 1
                    if cursor_train >= train_batch.size()[0]:
                        cursor_train = 0
                        train_batch, train_label = next(iter_tr)

            selected_train_pic = train_batch[cursor_train]

            from lime import lime_image
            import numpy as np

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(np.array(selected_pic.permute(1, 2, 0), dtype=np.double),
                                         batch_predict, # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function
            from skimage.segmentation import mark_boundaries
            test_pic, mask_test_pic = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

            explanation_train = explainer.explain_instance(np.array(selected_train_pic.permute(1, 2, 0), dtype=np.double),
                                         batch_predict, # classification function
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function
            train_pic, mask_train_pic = explanation_train.get_image_and_mask(explanation_train.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

            def vis_pic_trans(pic):
                pic = tc.tensor(pic).permute(2, 0, 1)
                invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
                pic = invTrans(pic.unsqueeze(0)).squeeze()
                return pic.permute(1, 2, 0).numpy()

            test_pic = mark_boundaries(vis_pic_trans(test_pic), mask_test_pic)
            train_pic = mark_boundaries(vis_pic_trans(train_pic), mask_train_pic)

            import matplotlib.pyplot as plt

            plt.imshow(train_pic)
            plt.savefig(ag.mode+"/train-"+str(i)+".png")
            plt.imshow(test_pic)
            plt.savefig(ag.mode+"/test-"+str(i)+".png")

    except (KeyboardInterrupt, SystemExit): pass

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--data_root", type = str, default = "./data/image_CLEF/")
    parser.add_argument("--traindom", type = str, default = "b")
    parser.add_argument("--testdoms", type = str, nargs = '+', default = ["b", "c", "i", "p"])

    parser.add_argument("--dim_s", type = int, default = 1024)
    parser.add_argument("--dim_v", type = int, default = 256)
    parser.add_argument("--dim_btnk", type = int, default = 1024) # for discr_model
    parser.add_argument("--dims_bb2bn", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2s", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_s2y", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2v", type = int, nargs = '*') # for discr_model
    parser.add_argument("--vbranch", type = boolstr, default = False) # for discr_model
    parser.add_argument("--dim_feat", type = int, default = 128) # for gen_model
    # parser.add_argument("--gpu", type=int, default=0)

    parser.set_defaults(discrstru = "ResNet50", genstru = "DCGANvar",
            n_bat = 100, n_epk = 100,
            optim = "SGD", lr = 4e-3, wl2 = 5e-4,
            momentum = .9, nesterov = True, lr_expo = .75, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
            wda = .25,
            domdisc_dimh = 1024, # for {dann, cdan, mdd} only
            cdan_rand = False, # for cdan only
            ker_alphas = [.5, 1., 2.], # for dan only
            mdd_margin = 4. # for mdd only
        )
    ag = parser.parse_args()
    bat_size = ag.n_bat
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    ag, ckpt = process_continue_run(ag)
    IS_OOD = is_ood(ag.mode)
    ag.n_bat = bat_size

    archtype = "cnn"
    # Dataset
    shape_x = (3, 224, 224) # determined by the loader
    dim_y = 12
    kwargs = {'num_workers': 4, 'pin_memory': True}
    tr_src_loader = data_loader.load_testing(
            ag.data_root, ag.traindom, ag.n_bat, kwargs) # needs to rand split otherwise some classes are unseen in training.
    val_src_loader = tr_src_loader
    ls_ts_tgt_loader = [data_loader.load_testing(
            ag.data_root, testdom, ag.n_bat, kwargs)
        for testdom in ag.testdoms]
    print(ag.testdoms, len(ls_ts_tgt_loader), len(tr_src_loader), len(ls_ts_tgt_loader[0]))
    if not IS_OOD:
        ls_tr_tgt_loader = [data_loader.load_testing(
            ag.data_root, testdom, ag.n_bat, kwargs)
            for testdom in ag.testdoms]

    if IS_OOD:
        get_visual( ag, ckpt, archtype, shape_x, dim_y,
                tr_src_loader, val_src_loader, ls_ts_tgt_loader )
    else:
        for testdom, tr_tgt_loader, ts_tgt_loader in zip(
                ag.testdoms, ls_tr_tgt_loader, ls_ts_tgt_loader):
            if testdom != ag.traindom:
                get_visual( ag, ckpt, archtype, shape_x, dim_y,
                        tr_src_loader, val_src_loader, None,
                        tr_tgt_loader, ts_tgt_loader, testdom )
            else:
                warnings.warn("same domain adaptation ignored")

