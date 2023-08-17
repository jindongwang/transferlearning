# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.margin_loss import LargeMarginLoss


class Fixed(Algorithm):

    def __init__(self, args):

        super(Fixed, self).__init__(args)

        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num-len(args.test_envs))
        self.args = args
        self.criterion = LargeMarginLoss(
            self.args.mixup_ld_margin, top_k=self.args.top_k, loss_type=self.args.ldmarginlosstype, reduce='none')

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_z = self.bottleneck(self.featurizer(all_x))

        disc_input = all_z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)

        disc_labels = torch.cat([data[2].cuda().long()
                                for data in minibatches])

        disc_loss = F.cross_entropy(disc_out, disc_labels)

        lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
        index2 = torch.randperm(all_z.shape[0]).cuda()

        all_mixz = lam*all_z+(1-lam)*all_z[index2]
        all_preds = self.classifier(all_mixz)
        classifier_loss = torch.mean(self.criterion(all_preds, all_y, [
                                     all_mixz])*lam+self.criterion(all_preds, all_y[index2], [all_mixz])*(1-lam))
        loss = classifier_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))
