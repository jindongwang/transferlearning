# encoding=utf-8
"""
    Created on 10:35 2018/12/29 
    @author: Jindong Wang
"""

import gzip
import pickle
from scipy.io import loadmat
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch


## For loading datasets of MNIST, USPS, and SVHN.


class GetDataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, data, label,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        if img.shape[0] != 1:
            # print(img)
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))
        #
        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            #  return img, target
        return img, target

    def __len__(self):
        return len(self.data)


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def load_mnist(path, scale=True, usps=False, all_use=True):
    mnist_data = loadmat(path)
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test = mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    if usps and all_use != 'yes':
        mnist_train = mnist_train[:2000]
        train_label = train_label[:2000]

    return mnist_train, train_label, mnist_test, test_label


def load_svhn(path_train, path_test):
    svhn_train = loadmat(path_train)
    svhn_test = loadmat(path_test)
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


def load_usps(path, all_use=True):
    f = gzip.open(path, 'rb')
    data_set = pickle.load(f, encoding='bytes')
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    if all_use == 'yes':
        img_train = img_train[inds][:6562]
        label_train = label_train[inds][:6562]
    else:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))
    return img_train, label_train, img_test, label_test


def load_dataset(domain, root_dir):
    train_img, train_label, test_img, test_label = None, None, None, None
    if domain == 'mnist':
        train_img, train_label, test_img, test_label = load_mnist(root_dir + 'mnist_data.mat')
    if domain == 'usps':
        train_img, train_label, test_img, test_label = load_usps(root_dir + 'usps_28x28.pkl')
    if domain == 'svhn':
        train_img, train_label, test_img, test_label = load_svhn(root_dir + 'svhn_train_32x32.mat',
                                                                 root_dir + 'svhn_test_32x32.mat')
    return train_img, train_label, test_img, test_label


def load_data(domain, root_dir, batch_size):
    src_train_img, src_train_label, src_test_img, src_test_label = load_dataset(domain['src'], root_dir)
    tar_train_img, tar_train_label, tar_test_img, tar_test_label = load_dataset(domain['tar'], root_dir)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_src_train, data_src_test = GetDataset(src_train_img, src_train_label,
                                               transform), GetDataset(src_test_img,
                                                                      src_test_label,
                                                                      transform)
    data_tar_train, data_tar_test = GetDataset(tar_train_img, tar_train_label,
                                               transform), GetDataset(tar_test_img,
                                                                      tar_test_label,
                                                                      transform)
    dataloaders = {}
    dataloaders['src'] = torch.utils.data.DataLoader(data_src_train, batch_size=batch_size, shuffle=True,
                                                     drop_last=False,
                                                     num_workers=4)
    dataloaders['val'] = torch.utils.data.DataLoader(data_src_test, batch_size=batch_size, shuffle=True,
                                                     drop_last=False,
                                                     num_workers=4)
    dataloaders['tar'] = torch.utils.data.DataLoader(data_tar_train, batch_size=batch_size, shuffle=True,
                                                     drop_last=False,
                                                     num_workers=4)
    return dataloaders
