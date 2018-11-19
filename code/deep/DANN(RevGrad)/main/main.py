import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from dataset.data_loader import GetLoader
from model import DANN

DATA_SRC, DATA_TAR = 'mnist', 'mnist_m'
IMG_DIR_SRC, IMG_DIR_TAR = '../dataset/mnist', '../dataset/mnist_m'
MODEL_ROOT = '../models'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
IMAGE_SIZE = 28
N_EPOCH = 100
LOG_INTERVAL = 20

result = [] # Save the results


def load_data():
    img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_source = datasets.MNIST(
        root=IMG_DIR_SRC,
        train=True,
        transform=img_transform,
        download=True
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8)
    train_list = IMG_DIR_TAR + '/mnist_m_train_labels.txt'
    dataset_target = GetLoader(
        data_root=IMG_DIR_TAR + '/mnist_m_train',
        data_list=train_list,
        transform=img_transform
    )
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8)
    return dataloader_source, dataloader_target


def load_test_data(dataset_name):
    img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if dataset_name == 'mnist_m':
        test_list = '../dataset/mnist_m/mnist_m_test_labels.txt'
        dataset = GetLoader(
            data_root='../dataset/mnist_m/mnist_m_test',
            data_list=test_list,
            transform=img_transform
        )
    else:
        dataset = datasets.MNIST(
            root=IMG_DIR_SRC,
            train=False,
            transform=img_transform,
            download=True
        )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    return dataloader


def test(model, dataset_name, epoch):
    alpha = 0
    dataloader = load_test_data(dataset_name)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = model(input_data=t_img, alpha=alpha)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / len(dataloader.dataset) * 100
    print('Epoch: [{}/{}], accuracy on {} dataset: {:.4f}%'.format(epoch, N_EPOCH, dataset_name, accu))
    return accu


def train(model, optimizer, dataloader_src, dataloader_tar):
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    for epoch in range(1, N_EPOCH + 1):
        model.train()
        len_dataloader = min(len(dataloader_src), len(dataloader_tar))
        data_src_iter = iter(dataloader_src)
        data_tar_iter = iter(dataloader_tar)

        i = 1
        while i < len_dataloader + 1:
            p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Training model using source data
            data_source = data_src_iter.next()
            optimizer.zero_grad()
            s_img, s_label = data_source[0].to(DEVICE), data_source[1].to(DEVICE)
            domain_label = torch.zeros(BATCH_SIZE).long().to(DEVICE)
            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img = data_target[0].to(DEVICE)
            domain_label = torch.ones(BATCH_SIZE).long().to(DEVICE)
            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_t_domain + err_s_domain

            err.backward()
            optimizer.step()

            if i % LOG_INTERVAL == 0:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], err_s_label: {:.4f}, err_s_domain: {:.4f}, err_t_domain: {:.4f}'.format(
                        epoch, N_EPOCH, i, len_dataloader, err_s_label.item(), err_s_domain.item(),
                        err_t_domain.item()))
            i += 1

        # Save model, and test using the source and target
        torch.save(model, '{}/mnist_mnistm_model_epoch_{}.pth'.format(MODEL_ROOT, epoch))
        acc_src = test(model, DATA_SRC, epoch)
        acc_tar = test(model, DATA_TAR, epoch)
        result.append([acc_src, acc_tar])
        np.savetxt('result.csv', np.array(result), fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    torch.random.manual_seed(100)
    loader_src, loader_tar = load_data()
    model = DANN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, optimizer, loader_src, loader_tar)
