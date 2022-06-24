"""
Provides some useful utils for torch model evaluation.
"""

from __future__ import absolute_import

import random

import PIL
import numpy as np
import torch
import torchvision

from evaldnn.utils import common


class ImageNetValDataset(torch.utils.data.Dataset):
    """ Class for loading and preprocessing imagenet validation set.

    One can download the imagenet validation set at http://image-net.org/.
    To use this class, one should also download ILSVRC2012_validation_ground_truth.txt
    and put it in the same directory as the imagenet validation set.

    Parameters
    ----------
    resize_size : integer
        Size used for resizing images.
    center_crop_size : integer
        Size used for center cropping images.
    preprocess : bool
        Indicate whether or not to preprocess the images, normalizing them with
        mean and standard deviation.
    shuffle : bool
        Indicate whether or not to shuffle the images.
    seed : integer
        Random seed used for shuffle.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, resize_size, center_crop_size, preprocess, shuffle=False, seed=None, num_max=None):
        self._preprocess = preprocess
        self._dir = common.user_home_dir() + '/EvalDNN-data/ILSVRC2012_img_val'
        with open(self._dir + '/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
            lines = f.readlines()
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(lines)
        if num_max is not None:
            lines = lines[:num_max]
        self._filenames = []
        self._y = []
        for line in lines:
            splits = line.split('---')
            if len(splits) != 5:
                continue
            self._filenames.append(splits[0])
            self._y.append(int(splits[2]))
        self._y = torch.LongTensor(self._y)
        self._transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(resize_size), torchvision.transforms.CenterCrop(center_crop_size), torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, index):
        path = self._dir + '/' + self._filenames[index]
        x = PIL.Image.open(path)
        x = x.convert('RGB')
        x = self._transforms(x)
        if self._preprocess:
            x = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(x)
        y = self._y[index]
        return x, y

    @property
    def filenames(self):
        return self._filenames


def imagenet_benchmark_zoo_model_names():
    """ Get the names of all models naturally supported by this toolbox.

    Returns
    -------
    list of str
        The names of all models supported.

    """
    return ['vgg16', 'vgg19', 'alexnet', 'densenet121',
            'densenet169', 'densenet201', 'googlenet',
            'inception_v3', 'mnasnet', 'mobilenet_v2',
            'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'shufflenet_V2',
            'squeezenet1_0', 'squeezenet1_1',
            'wide_resnet50_2', 'wide_resnet101_2']


def imagenet_benchmark_zoo(model_name, data_original_shuffle=True, data_original_seed=1997, data_original_num_max=None):
    """Get pretrained model, validation data and other relative info for evaluation.

    The method provides convenience for getting a pretrained model, validation data
    and other info needed for perform evaluation.
    With this method, one no longer needs to create model or preprocess the inputs
    on their own.

    Parameters
    ----------
    model_name : str
        Model name.
    data_original_shuffle : bool
        Indicate whether or not to shuffle original images.
    data_original_seed : integer
        Random seed used for shuffle original images.
    data_original_num_max : integer
        The maximum number of original images to load. If it is set to none, all images
        will be loaded.

    Returns
    -------
    model : instance of torch.nn.Module
        Pretrained model to evaluate.
    dataset_normalized: instance of torch.utils.data.Dataset
        Normalized dataset, used to do predictions and get intermediate outputs.
    dataset_original: instance of torch.utils.data.Dataset
        Original dataset, used to perform adversarial attack.
    preprocessing : tuple
        A tuple with two elements representing mean and standard deviation.
    num_classes : int
        The number of classes.
    bounds : tuple of length 2
        The bounds for the pixel values.

    """
    if model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet169':
        model = torchvision.models.densenet169(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet201':
        model = torchvision.models.densenet201(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=True)
        dataset_normalized = ImageNetValDataset(299, 299, True)
        dataset_original = ImageNetValDataset(299, 299, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mnasnet':
        model = torchvision.models.mnasnet1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'shufflenet_V2':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'squeezenet1_1':
        model = torchvision.models.squeezenet1_1(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'wide_resnet101_2':
        model = torchvision.models.wide_resnet101_2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    else:
        raise Exception('Invalid model name: ' + model_name + '. Available model names :' + str(imagenet_benchmark_zoo_model_names()))
    preprocessing = (np.array(ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(ImageNetValDataset.std).reshape((3, 1, 1)))
    num_classes = 1000
    bounds = (0, 1)
    return model, dataset_normalized, dataset_original, preprocessing, num_classes, bounds


def cifar10_dataset(train=False, num_max=None):
    """ Load cifar10 data.

    torchvision data utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    dataset = torchvision.datasets.CIFAR10(root=common.user_home_dir() + '/EvalDNN-data', train=train, download=True)
    x = torch.from_numpy(dataset.data / 255.0).permute(0, 3, 1, 2).float()
    y = torch.tensor(dataset.targets)
    if num_max is not None:
        x = x[:num_max]
        y = y[:num_max]
    dataset = torch.utils.data.dataset.TensorDataset(x, y)
    return dataset


def mnist_dataset(train=False, num_max=None):
    """ Load mnist data.

    torchvision data utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    dataset = torchvision.datasets.MNIST(root=common.user_home_dir() + '/EvalDNN-data', train=train, download=True)
    x = torch.unsqueeze((dataset.data / 255.0), 1)
    y = dataset.targets
    if num_max is not None:
        x = x[:num_max]
        y = y[:num_max]
    dataset = torch.utils.data.dataset.TensorDataset(x, y)
    return dataset
