import torch
import torch.nn.functional as F
import os
import math
import time
import pretty_errors

from deep_meda import DeepMEDA
import data_loader
from Config import *


kwargs = {'num_workers': 1, 'pin_memory': True}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_c, loss_m, mu = model(data_source, data_target, label_source)
        loss_mmd = (1-mu) * loss_m + mu * loss_c
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + param * lambd * loss_mmd
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print(f'Epoch: [{epoch:2d}/{num_iter}]\tLoss: {loss.item():.4f}\tcls_Loss: {loss_cls.item():.4f}\tl_mar: {loss_m.item():.4f}, l_con: {loss_c.item():.4f}\tmu: {mu:.2f}')

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim = 1), target).item() # sum up batch loss
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len_target_dataset
        print(f'{target_name} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len_target_dataset} ({100. * correct / len_target_dataset:.2f}%)')
    return correct


if __name__ == '__main__':
    model = DeepMEDA(num_classes=class_num).cuda()
    correct = 0
    model.cuda()
    time_start=time.time()
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
            torch.save(model, 'model.pkl')
        end_time = time.time()
        print(f'{source_name}-{target_name}: max correct: {correct} max accuracy: {100. * correct / len_target_dataset:.2f}%\n')
        print('cost time:', end_time - time_start)
