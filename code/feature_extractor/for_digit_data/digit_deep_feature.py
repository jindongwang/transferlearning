# encoding=utf-8
"""
    Created on 10:47 2018/12/29 
    @author: Jindong Wang
"""

from __future__ import print_function

import argparse

import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import copy
import digit_data_loader
import digit_network
import torchvision.transforms as transforms

# Command setting
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('-model', '-m', type=str, help='model name', default='resnet')
parser.add_argument('-batch_size', '-b', type=int, help='batch size', default=100)
parser.add_argument('-gpu', '-g', type=int, help='cuda id', default=0)
parser.add_argument('-source', '-src', type=str, default='mnist')
parser.add_argument('-target', '-tar', type=str, default='usps')
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
N_CLASS = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = {'src': int(args.batch_size), 'tar': int(args.batch_size)}
N_EPOCH = 100
MOMENTUM = 0.9
DECAY = 5e-4


def finetune(model, dataloaders, optimizer):
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, N_EPOCH + 1):
        for phase in ['src', 'val', 'tar']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).long()
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
            acc_hist.append([epoch_loss, epoch_acc])
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
                                                                                  epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'save_model/best_{}-{}-{}.pth'.format(args.source, args.target, epoch))
                np.savetxt('{}_{}_hist.csv'.format(args.source, args.target), np.asarray(acc_hist, dtype=float),
                           fmt='%.6f', delimiter=',')
        print()

    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    print('{}Best acc: {}'.format('*' * 10, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'save_model/best_{}_{}.pth'.format(args.source, args.target))
    print('Best model saved!')
    return model, best_acc, acc_hist


def extract_feature(model, model_path, dataloader, source, data_name):
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    fea = torch.zeros(1, 501).to(DEVICE)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            x = model.get_feature(inputs)
            x = x.view(x.size(0), -1)
            labels = labels.view(labels.size(0), 1).float()
            x = torch.cat((x, labels), dim=1)
            fea = torch.cat((fea, x), dim=0)
    fea_numpy = fea.cpu().numpy()
    np.savetxt('{}_{}.csv'.format(source, data_name), fea_numpy[1:], fmt='%.6f', delimiter=',')
    print('{} - {} done!'.format(source, data_name))


# You may want to use this function to simply classify them after getting features
def classify_1nn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    data = {'src': np.loadtxt(args.source + '_' + args.source + '.csv', delimiter=','),
            'tar': np.loadtxt(args.source + '_' + args.target + '.csv', delimiter=','),
            }
    Xs, Ys, Xt, Yt = data['src'][:, :-1], data['src'][:, -1], data['tar'][:, :-1], data['tar'][:, -1]
    Xs = StandardScaler(with_mean=0, with_std=1).fit_transform(Xs)
    Xt = StandardScaler(with_mean=0, with_std=1).fit_transform(Xt)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, Ys)
    ypred = clf.predict(Xt)
    acc = accuracy_score(y_true=Yt, y_pred=ypred)
    print('{} - {}: acc: {:.4f}'.format(args.source, args.target, acc))


if __name__ == '__main__':
    torch.manual_seed(10)
    # Load data
    root_dir = 'data/digit/'
    domain = {'src': str(args.source), 'tar': str(args.target)}
    dataloaders = digit_data_loader.load_data(domain, root_dir, args.batch_size)
    print(len(dataloaders['src'].dataset), len(dataloaders['val'].dataset))

    ## Load model
    model = digit_network.Network().to(DEVICE)
    print('Source:{}, target:{}'.format(domain['src'], domain['tar']))
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=DECAY)
    model_best, best_acc, acc_hist = finetune(model, dataloaders, optimizer)

    ## Extract features for the target domain
    model_path = 'save_model/best_{}_{}.pth'.format(args.source, args.target)
    extract_feature(model, model_path, dataloaders['tar'], args.source, args.target)

    ## If you want to extract features for the source domain, run the following lines ALONE by setting args.source=args.target
    # model_path = 'save_model/best_{}_{}.pth'.format(args.source, 'old target domain')
    # extract_feature(model, model_path, dataloaders['tar'], args.source, args.target)
