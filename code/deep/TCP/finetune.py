from IPython import embed
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import torch.utils.model_zoo as model_zoo
import mmd
import math
from tools import *


BATCH = 16
target_name = 'webcam'

class DANNet(nn.Module):
    def __init__(self):
        super(DANNet, self).__init__()
        model = models.vgg16(pretrained=True)  #False

        self.features = model.features
        for param in self.features.parameters(): #NOTE: prune:True  // finetune:False
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.cls_fc = nn.Linear(4096, 31)

    def forward(self, source, target):
        loss = 0
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        if self.training == True:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        return source, loss

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks_1 = {}
        self.filter_ranks_2 = {}

    def forward(self, x, x_target):  # NOTE: whether to add target data
        loss = 0
        self.activations1 = []
        self.activations2 = []
        self.gradients = []
        self.grad_index_1 = 0
        self.grad_index_2 = 0
        self.activation_to_layer_1 = {}
        self.activation_to_layer_2 = {}

        activation1_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank_1)
                self.activations1.append(x)
                self.activation_to_layer_1[activation1_index] = layer
                activation1_index += 1

        activation2_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x_target = module(x_target)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x_target.register_hook(self.compute_rank_2)
                self.activations2.append(x_target)
                self.activation_to_layer_2[activation2_index] = layer
                activation2_index += 1

        x = self.model.classifier(x.view(x.size(0), -1))
        x_target = self.model.classifier(x_target.view(x_target.size(0), -1))
        loss += mmd.mmd_rbf_noaccelerate(x, x_target)
        source_pred = self.model.cls_fc(x)

        return source_pred, loss


    def compute_rank_1(self, grad):
        activation1_index = len(self.activations1) - self.grad_index_1 - 1
        activation1 = self.activations1[activation1_index]
        values1 =  torch.sum((activation1 * grad), dim = 0).sum(dim=1).sum(dim=1)[:].data
        # Normalize the rank by the filter dimensions
        values1 = \
            values1 / (activation1.size(0) * activation1.size(2) * activation1.size(3))
        if activation1_index not in self.filter_ranks_1:
            self.filter_ranks_1[activation1_index] = \
                torch.FloatTensor(activation1.size(1)).zero_().cuda()
        self.filter_ranks_1[activation1_index] = values1
        self.grad_index_1 += 1

    def compute_rank_2(self, grad):
        activation2_index = len(self.activations2) - self.grad_index_2 - 1
        activation2 = self.activations2[activation2_index]
        values2 =  torch.sum((activation2 * grad), dim = 0).sum(dim=1).sum(dim=1)[:].data
        values2 = \
            values2 / (activation2.size(0) * activation2.size(2) * activation2.size(3))
        if activation2_index not in self.filter_ranks_2:
            self.filter_ranks_2[activation2_index] = \
                torch.FloatTensor(activation2.size(1)).zero_().cuda()
        self.filter_ranks_2[activation2_index] = values2
        self.grad_index_2 += 1


    def lowest_ranking_filters(self, num):
        data_1 = []
        for i in sorted(self.filter_ranks_1.keys()):
            for j in range(self.filter_ranks_1[i].size(0)):
                data_1.append((self.activation_to_layer_1[i], j, self.filter_ranks_1[i][j]))
        data_2 = []
        for i in sorted(self.filter_ranks_2.keys()):
            for j in range(self.filter_ranks_2[i].size(0)):
                data_2.append((self.activation_to_layer_2[i], j, self.filter_ranks_2[i][j]))
        data_3 = []
        data_3.extend(data_1)
        data_3.extend(data_2)
        dic = {}
        c = nsmallest(num*2, data_3, itemgetter(2))
        for i in range(len(c)):
            nm = str(c[i][0]) + '_' + str(c[i][1])
            if dic.get(nm)!=None:
                dic[nm] = min(dic[nm], c[i][2].item())
            else:
                dic[nm] = c[i][2].item()
        newc = []
        for i in range(len(list(dic.items()))):
            lyer = int(list(dic.items())[i][0].split('_')[0])
            filt = int(list(dic.items())[i][0].split('_')[1])
            val = torch.tensor(list(dic.items())[i][1])
            newc.append((lyer, filt, val))
        return nsmallest(num, newc, itemgetter(2))


    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks_1:
            v = torch.abs(self.filter_ranks_1[i])
            v = v / np.sqrt(torch.sum(v * v)).cuda()
            self.filter_ranks_1[i] = v.cpu()
        for i in self.filter_ranks_2:
            v = torch.abs(self.filter_ranks_2[i])
            v = v / np.sqrt(torch.sum(v * v)).cuda()
            self.filter_ranks_2[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PrunningFineTuner_VGGnet:
    def __init__(self, train_path, test_path, model):
        self.source_loader = dataset.loader(train_path)
        self.target_train_loader = dataset.loader(test_path)
        self.target_test_loader = dataset.test_loader(test_path)
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()
        self.len_source_loader = len(self.source_loader)
        self.len_target_loader = len(self.target_train_loader)
        self.len_source_dataset = len(self.source_loader.dataset)
        self.len_target_dataset = len(self.target_test_loader.dataset)
        self.max_correct = 0
        self.littlemax_correct = 0
        self.cur_model = None

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.target_test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            s_output, t_output = self.model(data, data)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= self.len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, self.len_target_dataset,
            100. * correct / self.len_target_dataset))
        return correct


    def train(self, optimizer = None, epoches = 10, save_name=None):
        for i in range(epoches):
            print("Epoch: ", i+1)
            self.train_epoch(optimizer, i+1, epoches+1)
            cur_correct = self.test()
            if cur_correct >= self.littlemax_correct:
                self.littlemax_correct = cur_correct
                self.cur_model = self.model
                print("write cur bset model")

            if cur_correct > self.max_correct:
                self.max_correct = cur_correct
                if save_name:
                    torch.save(self.model, str(save_name))
            print('amazon to webcam max correct: {} max accuracy{: .2f}%\n'.format(
                self.max_correct, 100.0 * self.max_correct / self.len_target_dataset))

        print("Finished fine tuning.")

    def train_epoch(self, optimizer = None, epoch = 0, epoches = 0, rank_filters = False):
        LEARNING_RATE = 0.01 / math.pow((1 + 10 * (epoch - 1) / epoches), 0.75) # 10*
        optimizer = torch.optim.SGD([
            {'params': self.model.features.parameters()},
            {'params': self.model.classifier.parameters()},
            {'params': self.model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 5, momentum=0.9, weight_decay=5e-4)

        iter_source = iter(self.source_loader)
        iter_target = iter(self.target_train_loader)
        self.model.train()

        for i in range(1, self.len_source_loader):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if len(data_target < BATCH):
                iter_target = iter(self.target_train_loader)
                data_target, _ = iter_target.next()
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
            data_source, label_source = Variable(data_source), Variable(label_source)
            data_target = Variable(data_target)
            self.model.zero_grad()
            if rank_filters:    # prune
                # add cls_loss and mmd_loss
                pred, loss_mmd = self.prunner.forward(data_source, data_target)
                loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), label_source)
                gamma = 2 / (1 + math.exp(-10 * (epoch) / epoches)) - 1
                loss = loss_cls + gamma * loss_mmd
                loss.backward()
                print('prune loss: {:.5f}  {:.5f}'.format(loss_cls.item(), loss_mmd.item()))
            else:
                label_source_pred, loss_mmd = self.model(data_source, data_target)
                loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
                gamma = 2 / (1 + math.exp(-10 * (epoch) / epoches)) - 1
                loss = loss_cls +  gamma * loss_mmd
                loss.backward()
                optimizer.step()
                if i % 50 == 0:
                    print('Train Epoch:{} [{}/{}({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                    epoch, i * len(data_source), self.len_source_dataset,
                        100. * i / self.len_source_loader, LEARNING_RATE, loss.item(), loss_cls.item(), loss_mmd.item()))


    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(epoch = 1, epoches = 10, rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self, perc_ind, perchan):
        #Get the accuracy before prunning
        self.test()

        self.model.train()
        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()  # the total nums of channels in convs
        num_filters_to_prune_per_iteration = perchan # 20
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        perc = perc_ind/10 # 2.0/10
        iterations = int(iterations * perc)  # set the percentage of prunning 80%

        print("Number of prunning iterations to reduce filters", iterations)
        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            if self.cur_model:
                print("load cur best")
                model = self.cur_model.cpu()
            else:
                model = self.model.cpu()

            for layer_index, filter_index in prune_targets:
                print(layer_index, filter_index)
                model = prune_vgg16_conv_layer(model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.littlemax_correct = 0
            self.train(optimizer, epoches = 5)  #10

        print("Finished. Going to fine tune the model a bit more")
        self.max_correct = 0
        self.train(optimizer, epoches = 20, save_name = "model_prunned_clsmmd_{:.1f}".format(perc))

def total_num_channels(model):
    filters = 0
    for name, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d):
             print(name, module, module.out_channels)
             filters = filters + module.out_channels
    print('total nums of channels in convs: %d'%(filters))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args.train_path = '/home/xxx/datasets/office_31/amazon'
    args.test_path = '/home/xxx/datasets/office_31/webcam'

    if args.train:
        model = DANNet().cuda()
    elif args.prune:
        #model = torch.load("./model_prunned_clsmmd_0.6").cuda()
        fine_tuner = PrunningFineTuner_VGGnet(args.train_path, args.test_path, model)
        fine_tuner.test()
        total_num_channels(model)
        print_model_parm_flops(model)
        print_model_parm_nums(model)
        #embed()

    fine_tuner = PrunningFineTuner_VGGnet(args.train_path, args.test_path, model)

    if args.train:
        #model = torch.load("./model_prunned_clsmmd_0.4").cuda()
        fine_tuner = PrunningFineTuner_VGGnet(args.train_path, args.test_path, model)
        fine_tuner.test()
        fine_tuner.train(epoches = 10,  save_name = 'model_t')

    elif args.prune:
        for perc_ind, perchan in zip([2.0], [32]):
            fine_tuner = PrunningFineTuner_VGGnet(args.train_path, args.test_path, model)
            fine_tuner.prune(perc_ind, perchan)

