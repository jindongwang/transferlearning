import argparse
import data_load
import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import copy

# Command setting
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--model_name', '-m', type=str, help='model name', default='alexnet')
parser.add_argument('--batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('--gpu', '-g', type=int, help='cuda id', default=0)
parser.add_argument('--source', '-src', type=str, default='amazon')
parser.add_argument('--target', '-tar', type=str, default='webcam')
parser.add_argument('--num_class', '-c', type=int, default=31)
parser.add_argument('--dataset_path', type=str, default='../../data/Office31/Original_images/')
parser.add_argument('--epoch', type=int, help='Train epochs', default=100)
parser.add_argument('--momentum', type=float, help='Momentum', default=.9)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = {'src': int(args.batchsize), 'tar': int(args.batchsize)}


def get_optimizer(model):
    learning_rate = args.lr
    param_group = []
    param_group += [{'params': model.base_network.parameters(), 'lr': learning_rate}]
    param_group += [{'params': model.classifier_layer.parameters(), 'lr': learning_rate * 10}]
    optimizer = optim.SGD(param_group, momentum=args.momentum)
    return optimizer


# Schedule learning rate according to DANN if you want to (while I think this equation is wierd therefore I did not use this one)
def lr_schedule(optimizer, epoch):
    def lr_decay(LR, n_epoch, e):
        return LR / (1 + 10 * e / n_epoch) ** 0.75

    for i in range(len(optimizer.param_groups)):
        if i < len(optimizer.param_groups) - 1:
            optimizer.param_groups[i]['lr'] = lr_decay(LEARNING_RATE, N_EPOCH, epoch)
        else:
            optimizer.param_groups[i]['lr'] = lr_decay(LEARNING_RATE, N_EPOCH, epoch) * 10


def finetune(model, dataloaders, optimizer, criterion, best_model_path, use_lr_schedule=False):
    N_EPOCH = args.epoch
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    
    for epoch in range(1, N_EPOCH + 1):
        if use_lr_schedule:
            lr_schedule(optimizer, epoch)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'train':
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
                torch.save(model.state_dict(), 'save_model/best_{}_{}-{}.pth'.format(args.model_name, args.source, epoch))
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    print('------Best acc: {}%'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), best_model_path)
    print('Best model saved!')
    return model, best_acc, acc_hist


# Extract features for given intermediate layers
# Currently, this only works for ResNet since AlexNet and VGGNET only have features and classifiers modules. 
# You will need to manually define a function in the forward function to extract features 
# (by letting it return features and labels). 
# Please follow digit_deep_network.py for reference.
class FeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.model = model._modules['module'] if type(model) == torch.nn.DataParallel else model
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.model._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def extract_feature(model, dataloader, save_path, load_from_disk=False, model_path=''):
    if load_from_disk:
        model = models.Network(base_net='alexnet', n_class=31)
        model.load_state_dict(torch.load(model_path)).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            feas = model.get_features(inputs)
            labels = labels.view(labels.size(0), 1).float()
            x = torch.cat((feas, labels), dim=1)
    fea_numpy = x.cpu().numpy()
    np.savetxt(save_path, fea_numpy[1:], fmt='%.6f', delimiter=',')

# You may want to classify with 1nn after getting features
def classify_1nn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    data = {'src': np.loadtxt('{}{}_{}.csv'.format(args.dataset, args.source, args.source), delimiter=','),
            'tar': np.loadtxt('{}{}_{}.csv'.format(args.dataset, args.source, args.target), delimiter=','),
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
    print('Loading data...')
    data_folder = args.dataset_path
    domain = {'src': str(args.source), 'tar': str(args.target)}
    dataloaders = {}
    data_test = data_load.load_data(data_folder+domain['tar'] + '/images', BATCH_SIZE['tar'], 'test')
    data_train = data_load.load_data(data_folder+domain['src'] + '/images/', BATCH_SIZE['src'], 'train', train_val_split=True, train_ratio=.8)
    dataloaders['train'], dataloaders['val'], dataloaders['test'] = data_train[0], data_train[1], data_test
    print('Data loaded: Source: {}, Target: {}'.format(args.source, args.target))

    # Finetune
    print('Begin fintuning...')
    net = models.Network(base_net=args.model_name, n_class=args.num_class).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net)
    save_path = 'save_model/best_{}_{}.pth'.format(args.model_name, args.source)
    model_best, best_acc, acc_hist = finetune(net, dataloaders, optimizer, criterion, save_path, use_lr_schedule=False)
    print('Finetune completed!')

    # Extract features for the target domain
    model_path = 'save_model/best_{}_{}.pth'.format(args.model_name, args.source)
    feature_save_path = '{}_{}.csv'.format(args.source, args.target)
    extract_feature(net, dataloaders['test'], feature_save_path)
    print('Deep features are extracted and saved!')
