import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import tqdm
import data_loader
from model import DANN
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--source', type=str, default='mnist')
parser.add_argument('--target', type=str, default='mnist_m')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--result_path', type=str, default='result/result.csv')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


def test(model, dataset_name, epoch):
    alpha = 0
    dataloader = data_loader.load_test_data(dataset_name)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = model(input_data=t_img, alpha=alpha)
            prob, pred = torch.max(class_output.data, 1)
            n_correct += (pred == t_label.long()).sum().item()

    acc = float(n_correct) / len(dataloader.dataset) * 100
    return acc


def train(model, optimizer, dataloader_src, dataloader_tar):
    loss_class = torch.nn.CrossEntropyLoss()
    best_acc = -float('inf')
    len_dataloader = min(len(dataloader_src), len(dataloader_tar))
    for epoch in range(args.nepoch):
        model.train()
        i = 1
        for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_src), enumerate(dataloader_tar)), total=len_dataloader, leave=False):
            _, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(
                DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            p = float(i + epoch * len_dataloader) / args.nepoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_output, err_s_domain = model(input_data=x_src, alpha=alpha)
            err_s_label = loss_class(class_output, y_src)
            _, err_t_domain = model(
                input_data=x_tar, alpha=alpha, source=False)
            err_domain = err_t_domain + err_s_domain
            err = err_s_label + args.gamma * err_domain
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            i += 1
        item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}'.format(
            epoch, args.nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item())
        print(item_pr)
        fp = open(args.result_path, 'a')
        fp.write(item_pr + '\n')

        # test
        acc_src = test(model, args.source, epoch)
        acc_tar = test(model, args.target, epoch)
        test_info = 'Source acc: {:.4f}, target acc: {:.4f}'.format(acc_src, acc_tar)
        fp.write(test_info + '\n')
        print(test_info)
        fp.close()
        
        if best_acc < acc_tar:
            best_acc = acc_tar
            if not os.path.exists(args.model_path):
                os.mkdirs(args.model_path)
            torch.save(model, '{}/mnist_mnistm.pth'.format(args.model_path))
    print('Test acc: {:.4f}'.format(best_acc))


if __name__ == '__main__':
    torch.random.manual_seed(10)
    loader_src, loader_tar = data_loader.load_data()
    model = DANN(DEVICE).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train(model, optimizer, loader_src, loader_tar)
