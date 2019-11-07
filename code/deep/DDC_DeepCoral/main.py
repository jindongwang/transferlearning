import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []

def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(CFG['epoch']):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        # Test
        test(model, target_test_loader)
    

def load_data(src, tar, root_dir):
    folder_src = root_dir + src + '/images/'
    folder_tar = root_dir + tar + '/images/'
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)

    source_name = "amazon"
    target_name = "webcam"

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, CFG['data_path'])

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)
