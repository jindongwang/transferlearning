# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class DIFEX(Algorithm):
    def __init__(self, args):
        super(DIFEX, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        self.tfbd = args.bottleneck//2

        self.teaf = get_fea(args)
        self.teab = common_network.feat_bottleneck(
            self.featurizer.in_features, self.tfbd, args.layer)
        self.teac = common_network.feat_classifier(
            args.num_classes, self.tfbd, args.classifier)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teab,
            self.teac
        )

    def teanettrain(self, dataloaders, epochs, opt1, sch1):
        self.teaNet.train()
        minibatches_iterator = zip(*dataloaders)
        for epoch in range(epochs):
            minibatches = [(tdata) for tdata in next(minibatches_iterator)]
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])
            all_z = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            all_y = torch.cat([data[1].cuda().long() for data in minibatches])
            all_p = self.teaNet(all_z)
            loss = F.cross_entropy(all_p, all_y, reduction='mean')
            opt1.zero_grad()
            loss.backward()
            if ((epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.7)) == 0 or (epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.9)) == 0) and (not self.args.schuse):
                for param_group in opt1.param_groups:
                    param_group['lr'] = param_group['lr']*0.1
            opt1.step()
            if sch1:
                sch1.step()

            if epoch % int(self.args.steps_per_epoch) == 0 or epoch == epochs-1:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        self.teaNet.eval()

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        with torch.no_grad():
            all_x1 = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            tfea = self.teab(self.teaf(all_x1)).detach()

        all_z = self.bottleneck(self.featurizer(all_x))
        loss1 = F.cross_entropy(self.classifier(all_z), all_y)

        loss2 = F.mse_loss(all_z[:, :self.tfbd], tfea)*self.args.alpha
        if self.args.disttype == '2-norm':
            loss3 = -F.mse_loss(all_z[:, :self.tfbd],
                                all_z[:, self.tfbd:])*self.args.beta
        elif self.args.disttype == 'norm-2-norm':
            loss3 = -F.mse_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                                all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'norm-1-norm':
            loss3 = -F.l1_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                               all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'cos':
            loss3 = torch.mean(F.cosine_similarity(
                all_z[:, :self.tfbd], all_z[:, self.tfbd:]))*self.args.beta
        loss4 = 0
        if len(minibatches) > 1:
            for i in range(len(minibatches)-1):
                for j in range(i+1, len(minibatches)):
                    loss4 += self.coral(all_z[i*self.args.batch_size:(i+1)*self.args.batch_size, self.tfbd:],
                                        all_z[j*self.args.batch_size:(j+1)*self.args.batch_size, self.tfbd:])
            loss4 = loss4*2/(len(minibatches) *
                             (len(minibatches)-1))*self.args.lam
        else:
            loss4 = self.coral(all_z[:self.args.batch_size//2, self.tfbd:],
                               all_z[self.args.batch_size//2:, self.tfbd:])
            loss4 = loss4*self.args.lam

        loss = loss1+loss2+loss3+loss4
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss1.item(), 'dist': (loss2).item(), 'exp': (loss3).item(), 'align': loss4.item(), 'total': loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))
