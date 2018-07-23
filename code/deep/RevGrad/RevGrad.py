from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings
batch_size = 32
epochs = 200
lr = 0.01
momentum = 0.9
no_cuda =False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./dataset/"
source_name = "amazon"
target_name = "webcam"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k and not "domain_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

def train(epoch, model):
    #最后的全连接层学习率为前面的10倍
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print("learning rate：", LEARNING_RATE)
    optimizer_fea = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    optimizer_critic = torch.optim.SGD([
        {'params': model.domain_fc.parameters(), 'lr': LEARNING_RATE}
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)
    dlabel_src = Variable(torch.ones(batch_size).long().cuda())
    dlabel_tgt = Variable(torch.zeros(batch_size).long().cuda())
    i = 1
    while i <= len_source_loader:
        model.train()

        source_data, source_label = data_source_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        clabel_src, dlabel_pred_src = model(source_data)
        label_loss = F.nll_loss(F.log_softmax(clabel_src, dim=1), source_label)
        critic_loss_src = F.nll_loss(F.log_softmax(dlabel_pred_src, dim=1), dlabel_src)
        confusion_loss_src = 0.5 * ( F.nll_loss(F.log_softmax(dlabel_pred_src, dim=1), dlabel_src) + F.nll_loss(F.log_softmax(dlabel_pred_src, dim=1), dlabel_tgt) )

        target_data, target_label = data_target_iter.next()
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)
        clabel_tgt, dlabel_pred_tgt = model(target_data)
        critic_loss_tgt = F.nll_loss(F.log_softmax(dlabel_pred_tgt, dim=1), dlabel_tgt)
        confusion_loss_tgt = 0.5 * (F.nll_loss(F.log_softmax(dlabel_pred_tgt, dim=1), dlabel_src) + F.nll_loss(
            F.log_softmax(dlabel_pred_tgt, dim=1), dlabel_tgt))

        confusion_loss_total = (confusion_loss_src + confusion_loss_tgt) / 2
        fea_loss_total = confusion_loss_total + label_loss
        critic_loss_total = (critic_loss_src + critic_loss_tgt) / 2

        optimizer_fea.zero_grad()
        fea_loss_total.backward(retain_graph=True)
        optimizer_fea.step()
        optimizer_fea.zero_grad()
        optimizer_critic.zero_grad()
        critic_loss_total.backward()
        optimizer_critic.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tconfusion_Loss: {:.6f}\tlabel_Loss: {:.6f}\tdomain_Loss: {:.6f}'.format(
                epoch, i * len(source_data),len_source_dataset,
                100. * i / len_source_loader, confusion_loss_total.data[0], label_loss.data[0], critic_loss_total.data[0]))
        i = i + 1

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output, t_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return correct


if __name__ == '__main__':
    model = models.RevGrad(num_classes=31)
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain(model)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, target_name, correct, 100. * correct / len_target_dataset ))