"""
Provides some useful utils for mxnet model evaluation.
"""

from __future__ import absolute_import

import random

import gluoncv
import mxnet
import numpy as np

from evaldnn.utils import common


class ImageNetValDataset(mxnet.gluon.data.Dataset):
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
        self._y = mxnet.nd.array(self._y, dtype=int)
        self._transform = mxnet.gluon.data.vision.transforms.Compose([mxnet.gluon.data.vision.transforms.Resize(resize_size, keep_ratio=True), mxnet.gluon.data.vision.transforms.CenterCrop(center_crop_size), mxnet.gluon.data.vision.transforms.ToTensor()])

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, index):
        path = self._dir + '/' + self._filenames[index]
        x = mxnet.image.imread(path)
        x = self._transform(x)
        if self._preprocess:
            mean = mxnet.nd.array(self.mean).reshape(3, 1, 1)
            std = mxnet.nd.array(self.std).reshape(3, 1, 1)
            x = (x - mean) / std
        y = self._y[index].asscalar()
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
            'densenet169', 'densenet201', 'mobilenet0_25',
            'mobilenet0_5', 'mobilenet1_0', 'mobilenet_v2_1_0',
            'resnet101_v1', 'resnet101_v2', 'resnet152_v1',
            'resnet152_v2', 'resnet50_v1', 'resnet50_v2',
            'resnext50_32x4d', 'squeezenet1_0',
            'inception_v3', 'xception']


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
    model : instance of mxnet.gluon.nn.Block
        Pretrained model to evaluate.
    dataset_normalized: instance of mxnet.gluon.data.Dataset
        Normalized dataset, used to do predictions and get intermediate outputs.
    dataset_original: instance of mxnet.gluon.data.Dataset
        Original dataset, used to perform adversarial attack.
    preprocessing : tuple
        A tuple with two elements representing mean and standard deviation.
    num_classes : int
        The number of classes.
    bounds : tuple of length 2
        The bounds for the pixel values.

    """
    if model_name == 'vgg16':
        model = gluoncv.model_zoo.vgg16(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'vgg19':
        model = gluoncv.model_zoo.vgg19(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'alexnet':
        model = gluoncv.model_zoo.alexnet(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet121':
        model = gluoncv.model_zoo.densenet121(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet169':
        model = gluoncv.model_zoo.densenet169(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'densenet201':
        model = gluoncv.model_zoo.densenet201(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet0_25':
        model = gluoncv.model_zoo.mobilenet0_25(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet0_5':
        model = gluoncv.model_zoo.mobilenet0_5(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet1_0':
        model = gluoncv.model_zoo.mobilenet1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v2_1_0':
        model = gluoncv.model_zoo.mobilenet_v2_1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet101_v1':
        model = gluoncv.model_zoo.resnet101_v1(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet101_v2':
        model = gluoncv.model_zoo.resnet101_v2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet152_v1':
        model = gluoncv.model_zoo.resnet152_v1(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet152_v2':
        model = gluoncv.model_zoo.resnet152_v2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet50_v1':
        model = gluoncv.model_zoo.resnet50_v1(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet50_v2':
        model = gluoncv.model_zoo.resnet50_v2(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnext50_32x4d':
        model = gluoncv.model_zoo.get_model('ResNext50_32x4d', pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'squeezenet1_0':
        model = gluoncv.model_zoo.squeezenet1_0(pretrained=True)
        dataset_normalized = ImageNetValDataset(256, 224, True)
        dataset_original = ImageNetValDataset(256, 224, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v3':
        model = gluoncv.model_zoo.inception_v3(pretrained=True)
        dataset_normalized = ImageNetValDataset(299, 299, True)
        dataset_original = ImageNetValDataset(299, 299, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'xception':
        model = gluoncv.model_zoo.get_model('Xception', pretrained=True)
        dataset_normalized = ImageNetValDataset(299, 299, True)
        dataset_original = ImageNetValDataset(299, 299, False, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    else:
        raise Exception('Invalid model name: ' + model_name + '. Available model names :' + str(imagenet_benchmark_zoo_model_names()))
    preprocessing = (np.array(ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(ImageNetValDataset.std).reshape((3, 1, 1)))
    num_classes = 1000
    bounds = (0, 1)
    return model, dataset_normalized, dataset_original, preprocessing, num_classes, bounds


def cifar10_dataset(train=False, num_max=None):
    """ Load cifar10 data.

    Gluon data utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    dataset = mxnet.gluon.data.vision.CIFAR10(train=train)
    dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
    if num_max is not None:
        dataset._data = dataset._data[:num_max]
        dataset._label = dataset._label[:num_max]
    return dataset


def mnist_dataset(train=False, num_max=None):
    """ Load mnist data.

    Gluon data utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    dataset = mxnet.gluon.data.vision.MNIST(train=train)
    dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
    if num_max is not None:
        dataset._data = dataset._data[:num_max]
        dataset._label = dataset._label[:num_max]
    return dataset
