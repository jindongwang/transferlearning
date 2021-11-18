# coding=utf-8
from torchvision import datasets, transforms
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io


class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class myDataset(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        class_dirs = [os.path.join(root, i) for i in os.listdir(root)]
        imgs = []
        for i in class_dirs:
            imgs += [os.path.join(i, img) for img in os.listdir(i)]
        np.random.shuffle(imgs)
        imgs_mun = len(imgs)
        # target:val = 8 ：2
        if self.train:
            self.imgs = imgs[:int(0.3*imgs_mun)]
        else:
            self.imgs = imgs[int(0.3*imgs_mun):]
        if transform:
            self.transforms = transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            start_center = (256 - 224 - 1) / 2
            self.transforms = transforms.Compose(
                [transforms.Resize([224, 224]),
                 PlaceCrop(224, start_center, start_center),
                 transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = int(img_path.strip().split('/')[10])
        print(img_path, label)
        #data = Image.open(img_path)
        data = io.imread(img_path)
        data = Image.fromarray(data)
        if data.getbands()[0] == 'L':
            data = data.convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


def load_training(root_path, domain, batch_size, kwargs, train_val_split=.5, rand_split=True):
    kwargs_fin = dict(shuffle=True, drop_last=True)
    kwargs_fin.update(kwargs)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [ResizeImage(256),
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize])
    data = datasets.ImageFolder(root=os.path.join(
        root_path, domain), transform=transform)
    if train_val_split <= 0:
        train_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, **kwargs_fin)
        return train_loader
    else:
        train_loader, val_loader = load_train_valid_split(
            data, batch_size, kwargs_fin, val_ratio=1.-train_val_split, rand_split=rand_split)
        return train_loader, val_loader


def load_testing(root_path, domain, batch_size, kwargs):
    kwargs_fin = dict(shuffle=False, drop_last=False)
    kwargs_fin.update(kwargs)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [ResizeImage(256),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor(),
         normalize])
    dataset = datasets.ImageFolder(root=os.path.join(
        root_path, domain), transform=transform)
    test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, **kwargs_fin)
    return test_loader


def load_train_valid_split(dataset, batch_size, kwargs, val_ratio=0.4, rand_split=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if rand_split: np.random.shuffle(indices)
    len_val = int(np.floor(val_ratio * dataset_size))
    train_indices, val_indices = indices[len_val:], indices[:len_val]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    __ = kwargs.pop('shuffle', None)
    __ = kwargs.pop('drop_last', None)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, **kwargs, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, **kwargs, drop_last=True)
    return train_loader, validation_loader

def load_data(root_path, source_dir, target_dir, batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    source_loader = load_training(
        root_path, source_dir, batch_size, kwargs)
    target_loader = load_training(
        root_path, target_dir, batch_size, kwargs)
    test_loader = load_testing(
        root_path, target_dir, batch_size, kwargs)
    return source_loader, target_loader, test_loader

def load_all_test(root_path, dataset, batch_size, train, kwargs):
    ls = []
    domains = {'Office-31': ['amazon', 'dslr', 'webcam'],
            'Office-Home': ['Art', 'Clipart', 'Product', 'RealWorld']}
    for dom in domains[dataset]:
        if train:
            loader = load_training(root_path, dom, batch_size, kwargs, train_val_split=-1)
        else:
            loader = load_testing(root_path, dom, batch_size, kwargs)
        ls.append(loader)
    return ls
