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

# Command setting
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('-model', '-m', type=str, help='model name', default='resnet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=int, help='cuda id', default=0)
parser.add_argument('-source', '-src', type=str, default='amazon')
parser.add_argument('-target', '-tar', type=str, default='webcam')
parser.add_argument('-num_class', '-c', type=int, default=12)
parser.add_argument('-dataset', type=str, default='visda')
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
N_CLASS = args.num_class
LEARNING_RATE = 1e-4
BATCH_SIZE = {'src': int(args.batchsize), 'tar': int(args.batchsize)}
N_EPOCH = 100
MOMENTUM = 0.9
DECAY = 0

def load_model(name='resnet'):
    model = None
    if name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        n_features = model.classifier[6].in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.classifier[6] = fc
    elif name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        n_features = model.classifier[6].in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.classifier[6] = fc
    elif name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        n_features = model.fc.in_features
        fc = torch.nn.Linear(n_features, N_CLASS)
        model.fc = fc
    return model


def get_optimizer(model_name):
    learning_rate = LEARNING_RATE
    if model_name == 'alexnet':
        param_group = [{'params': model.features.parameters(), 'lr': learning_rate}]
        for i in range(6):
            param_group += [{'params': model.classifier[i].parameters(), 'lr': learning_rate}]
        param_group += [{'params': model.classifier[6].parameters(), 'lr': learning_rate * 10}]
    elif model_name == 'resnet':
        param_group = []
        for k, v in model.named_parameters():
            if not k.__contains__('fc'):
                param_group += [{'params': v, 'lr': learning_rate}]
            else:
                param_group += [{'params': v, 'lr': learning_rate * 10}]
    optimizer = optim.SGD(param_group, momentum=MOMENTUM, weight_decay=DECAY)
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


def finetune(model, dataloaders, optimizer):
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(31, N_EPOCH + 1):
        # lr_schedule(optimizer, epoch)
        # print('Learning rate: {:.8f}'.format(optimizer.param_groups[0]['lr']))
        # print('Learning rate: {:.8f}'.format(optimizer.param_groups[-1]['lr']))
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
            acc_hist.append([epoch_loss, epoch_acc])
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
                                                                                  epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'save_model/best_{}_{}-{}.pth'.format(args.model, args.source, epoch))
        print()
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    print('{}Best acc: {}'.format('*' * 10, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'save_model/best_{}_{}.pth'.format(args.model, args.source))
    print('Best model saved!')
    return model, best_acc, acc_hist


# Extract features for given intermediate layers
# Currently, this only works for ResNet since AlexNet and VGGNET only have features and classifiers modules. You will need to manually define a function in the forward function to extract features (by letting it return features and labels). Please follow digit_deep_network.py for reference.
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


def extract_feature(model_name, model_path, dataloader, source, data_name):
    model = load_model(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    extract_list = ['avgpool'] # ResNet
    fea = torch.zeros(1, 2049).to(DEVICE)
    myextractor = FeatureExtractor(model, extract_list)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            x = myextractor(inputs)[0]
            x = x.view(x.size(0), -1)
            labels = labels.view(labels.size(0), 1).float()
            x = torch.cat((x, labels), dim=1)
            fea = torch.cat((fea, x), dim=0)
    fea_numpy = fea.cpu().numpy()
    np.savetxt('{}_{}.csv'.format(source, data_name), fea_numpy[1:], fmt='%.6f', delimiter=',')
    print('{} - {} done!'.format(source, data_name))

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

# Configure you own dataset folder
def dataset():
    if args.dataset == 'office31' or 'office-31':
        return 'data/OFFICE31/'
    elif args.dataset == 'officehome' or 'office_home' or 'office-home':
        return 'data/OfficeHome/'
    elif args.dataset == 'visda':
        return 'data/VisDA17/'
    elif args.dataset == 'imageclef' or 'image-clef':
        return 'data/image_CLEF'


if __name__ == '__main__':
    torch.manual_seed(10)

    # Load data
    root_dir = dataset()
    domain = {'src': str(args.source), 'tar': str(args.target)}
    dataloaders = {}
    dataloaders['tar'] = data_loader.load_data(root_dir, domain['tar'], BATCH_SIZE['tar'], 'tar')
    dataloaders['src'], dataloaders['val'] = data_loader.load_train(root_dir, domain['src'], BATCH_SIZE['src'], 'src')
    print(len(dataloaders['src'].dataset), len(dataloaders['val'].dataset))

    ## Finetune
    model = load_model(model_name).to(DEVICE)
    print('Source:{}, target:{}, model: {}'.format(domain['src'], domain['tar'], model_name))
    optimizer = get_optimizer(model_name)
    model_best, best_acc, acc_hist = finetune(model, dataloaders, optimizer)

    ## Extract features for the target domain
    model_path = 'save_model/best_{}_{}.pth'.format(args.model, args.source)
    extract_feature(args.model, model_path, dataloaders['tar'], args.source, args.target)
    # If you want to extract features for source domain, you can run the following line ALONE by setting both args.source and args.target = source domain
    # extract_feature(args.model, model_path, dataloaders['tar'], args.source, args.target)

