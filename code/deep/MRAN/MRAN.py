from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=3, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="../OFFICE31/",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="webcam",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="dslr",
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)

    return source_train_loader, target_train_loader, target_test_loader

def train(epoch, model, source_loader, target_loader):
    #最后的全连接层学习率为前面的10倍
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
    print("learning rate：", LEARNING_RATE)
    if args.diff_lr:
        optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.Inception.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)
    model.train()
    tgt_iter = iter(target_loader)
    for batch_idx, (source_data, source_label) in enumerate(source_loader):
        try:
            target_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter=iter(target_loader)
            target_data, _ = tgt_iter.next()
        
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        optimizer.zero_grad()

        s_output, mmd_loss = model(source_data, target_data, source_label)
        soft_loss = F.nll_loss(F.log_softmax(s_output, dim=1), source_label)
        # print((2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1))
        if args.gamma == 1:
            gamma = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
        if args.gamma == 2:
            gamma = epoch /args.epochs
        loss = soft_loss + gamma * mmd_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlabel_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, batch_idx * len(source_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), soft_loss.item(), mmd_loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            s_output, t_output = model(data, data, target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, reduction='sum').item()# sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


if __name__ == '__main__':
    model = models.MRANNet(num_classes=31)
    print(model)
    if args.cuda:
        model.cuda()
    train_loader, unsuptrain_loader, test_loader = load_data()
    correct = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, unsuptrain_loader)
        t_correct = test(model, test_loader)
        if t_correct > correct:
            correct = t_correct
        print("%s max correct:" % args.test_dir, correct.item())
        print(args.source_dir, "to", args.test_dir)
