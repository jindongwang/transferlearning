import torch
import os
import math
import data_loader
import models
from config import CFG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(CFG['epoch']):
        # Train
        model.train()
        model.isTrain = True
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
            label_source_pred, loss_coral = model(data_source, data_target)
            loss_cls = criterion(label_source_pred, label_source)
            loss = loss_cls + CFG['lambda'] * loss_coral
            loss.backward()
            optimizer.step()
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\t \
                    total_Loss: {:.6f}\t \
                    cls_Loss: {:.6f}\t \
                    coral_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_coral.item()))
       
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(target_test_loader.dataset)
        with torch.no_grad():
            model.isTrain = False
            for data, target in target_test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                s_output, _ = model(data, None)
                test_loss += criterion(s_output, target)
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target.data)

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_target_dataset))


def load_data(src, tar, root_dir):
    source_loader = data_loader.load_data(
        root_dir, src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        root_dir, tar, CFG['batch_size'], False, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        root_dir, tar, CFG['batch_size'], False, CFG['kwargs'])            

if __name__ == '__main__':
    torch.manual_seed(CFG['seed'])

    source_name = "amazon"
    target_name = "webcam"

    source_loader, target_train_loader, target_test_loader = load_data(source_name, target_name, CFG['data_path'])

    model = models.DeepCoral(CFG['n_class'], CFG['backbone']).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.fc.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG)
