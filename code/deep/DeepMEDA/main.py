import torch
import torch.nn.functional as F
import math
import pretty_errors
import argparse
import numpy as np

from deep_meda import DeepMEDA
import data_loader


def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders):
    LEARNING_RATE = args.lr / \
        math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.decay)

    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_c, loss_m, mu = model(
            data_source, data_target, label_source)
        loss_mmd = (1-mu) * loss_m + mu * loss_c
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_mmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}/{num_iter}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, l_mar: {loss_m.item():.4f}, l_con: {loss_c.item():.4f}, mu: {mu:.2f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader)} ({100. * correct / len(dataloader):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/data/jindongwang/office31/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='dslr')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='amazon')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.3)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    model = DeepMEDA(num_classes=args.nclass).cuda()
    correct = 0
    stop = 0
    for epoch in range(1, args.nepoch + 1):
        stop += 1
        train_epoch(epoch, model, dataloaders)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break
