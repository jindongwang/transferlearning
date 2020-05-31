import torch.utils.data as data
import torch
from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets

DATA_SRC, DATA_TAR = 'mnist', 'mnist_m'
IMG_DIR_SRC, IMG_DIR_TAR = 'dataset/mnist', 'dataset/mnist_m/mnist_m'
IMAGE_SIZE = 28
BATCH_SIZE = 128

img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

def load_data(src_root=IMG_DIR_SRC, tar_root=IMG_DIR_TAR, batch_size=128):
    
    dataset_source = datasets.MNIST(
        root=src_root,
        train=True,
        transform=img_transform,
        download=True
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)

    train_list = os.path.join(tar_root, 'mnist_m_train_labels.txt')
    dataset_target = GetLoader(
        data_root=tar_root + '/mnist_m_train',
        data_list=train_list,
        transform=img_transform
    )
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8)
    return dataloader_source, dataloader_target


def load_test_data(dataset_name, batch_size=128):
    if dataset_name == 'mnist_m':
        test_list = 'dataset/mnist_m/mnist_m/mnist_m_test_labels.txt'
        dataset = GetLoader(
            data_root='dataset/mnist_m/mnist_m/mnist_m_test',
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

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
