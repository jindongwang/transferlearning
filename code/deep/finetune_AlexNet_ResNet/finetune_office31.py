from __future__ import print_function

import argparse

import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time


# Command setting
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--src', type=str, default='amazon')
parser.add_argument('--tar', type=str, default='webcam')
parser.add_argument('--n_class', type=int, default=31)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--data', type=str, default='Your dataset folder')
parser.add_argument('--early_stop', type=int, default=20)
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda')
BATCH_SIZE = {'src': int(args.batchsize), 'tar': int(args.batchsize)}


def load_model(name='alexnet'):
    if name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        n_features = model.classifier[6].in_features
        fc = torch.nn.Linear(n_features, args.n_class)
        model.classifier[6] = fc
    elif name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        n_features = model.fc.in_features
        fc = torch.nn.Linear(n_features, args.n_class)
        model.fc = fc
    model.fc.weight.data.normal_(0, 0.005)
    model.fc.bias.data.fill_(0.1)
    return model


def get_optimizer(model_name):
    learning_rate = args.lr
    if model_name == 'alexnet':
        param_group = [
            {'params': model.features.parameters(), 'lr': learning_rate}]
        for i in range(6):
            param_group += [{'params': model.classifier[i].parameters(),
                             'lr': learning_rate}]
        param_group += [{'params': model.classifier[6].parameters(),
                         'lr': learning_rate * 10}]
    elif model_name == 'resnet':
        param_group = []
        for k, v in model.named_parameters():
            if not k.__contains__('fc'):
                param_group += [{'params': v, 'lr': learning_rate}]
            else:
                param_group += [{'params': v, 'lr': learning_rate * 10}]
    optimizer = optim.SGD(param_group, momentum=args.momentum)
    return optimizer


# Schedule learning rate
def lr_schedule(optimizer, epoch):
    def lr_decay(LR, n_epoch, e):
        return LR / (1 + 10 * e / n_epoch) ** 0.75

    for i in range(len(optimizer.param_groups)):
        if i < len(optimizer.param_groups) - 1:
            optimizer.param_groups[i]['lr'] = lr_decay(
                args.lr, args.n_epoch, epoch)
        else:
            optimizer.param_groups[i]['lr'] = lr_decay(
                args.lr, args.n_epoch, epoch) * 10

def test(model, target_test_loader):
    model.eval()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model(data)
            loss = criterion(s_output, target)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = correct.double() / len(target_test_loader.dataset)
    return acc

def finetune(model, dataloaders, optimizer):
    since = time.time()
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    stop = 0
    for epoch in range(1, args.n_epoch + 1):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['src', 'val', 'tar']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders[phase].dataset)
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, args.n_epoch, phase, epoch_loss,
                                                                                  epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'model.pkl')
        if stop >= args.early_stop:
            break
        print()
    model.load_state_dict(torch.load('model.pkl'))
    acc_test = test(model, dataloaders['tar'])
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    return model, acc_test


if __name__ == '__main__':
    torch.manual_seed(10)
    # Load data
    root_dir = args.data
    domain = {'src': str(args.src), 'tar': str(args.tar)}
    dataloaders = {}
    dataloaders['tar'] = data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'tar')
    dataloaders['src'], dataloaders['val'] = data_loader.load_train(root_dir, domain['src'], BATCH_SIZE['src'], 'src')
    # Load model
    model_name = str(args.model)
    model = load_model(model_name).to(DEVICE)
    print('Source: {} ({}), target: {} ({}), model: {}'.format(
        domain['src'], len(dataloaders['src'].dataset), domain['tar'], len(dataloaders['val'].dataset), model_name))
    optimizer = get_optimizer(model_name)
    model_best, best_acc = finetune(model, dataloaders, optimizer)
    print('Best acc: {}'.format(best_acc))
