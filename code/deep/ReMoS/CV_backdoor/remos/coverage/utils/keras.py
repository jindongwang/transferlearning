"""
Provides some useful utils for keras model evaluation.
"""

from __future__ import absolute_import

import random

import cv2
import keras
import numpy as np

from evaldnn.utils import common


class ImageNetValData():
    """ Class for loading and preprocessing imagenet validation set.

    One can download the imagenet validation set at http://image-net.org/.
    To use this class, one should also download ILSVRC2012_validation_ground_truth.txt
    and put it in the same directory as the imagenet validation set.

    Parameters
    ----------
    fashion : str
        Indicate the preprocessing fashion. It can be either vgg_preprocessing
        or inception_preprocessing.
    size : integer
        Target image size (input size).
    transform : function(image) -> image
        The transform function for preprocessing image after the image is loaded
        and cropped to proper size.
    shuffle : bool
        Indicate whether or not to shuffle the images.
    seed : integer
        Random seed used for shuffle.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """

    class ImageNetValDataX():

        def __init__(self, dir, filenames, fashion, size, transform):
            self._dir = dir
            self._filenames = filenames
            self._fashion = fashion
            self._size = size
            self._transform = transform

        def __len__(self):
            return len(self._filenames)

        def __getitem__(self, index):
            x = None
            for filename in self._filenames[index]:
                path = self._dir + '/' + filename
                image = cv2.imread(path)
                if self._fashion == 'vgg_preprocessing':
                    height, width, _ = image.shape
                    new_height = height * 256 // min(image.shape[:2])
                    new_width = width * 256 // min(image.shape[:2])
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    height, width, _ = image.shape
                    startx = width // 2 - (self._size // 2)
                    starty = height // 2 - (self._size // 2)
                    image = image[starty:starty + self._size, startx:startx + self._size]
                elif self._fashion == 'inception_preprocessing':
                    height, width, channels = image.shape
                    assert channels == 3
                    new_height = int(height * 0.875)
                    new_width = int(width * 0.875)
                    startx = (width - new_width) // 2
                    starty = (height - new_height) // 2
                    image = image[starty:starty + new_height, startx:startx + new_width]
                    image = cv2.resize(image, (self._size, self._size), interpolation=cv2.INTER_CUBIC)
                else:
                    raise Exception('Unknown fashion', self._fashion)
                image = image[:, :, ::-1]
                if self._transform is not None:
                    image = self._transform(image)
                image = np.expand_dims(image, axis=0)
                if x is None:
                    x = image
                else:
                    x = np.concatenate((x, image), axis=0)
            x = x.astype(np.float32)
            return x

    def __init__(self, fashion, size, transform=None, shuffle=False, seed=None, num_max=None):
        dir = common.user_home_dir() + '/EvalDNN-data/ILSVRC2012_img_val'
        with open(dir + '/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
            lines = f.readlines()
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(lines)
        if num_max is not None:
            lines = lines[:num_max]
        self._filenames = []
        self.y = []
        for line in lines:
            splits = line.split('---')
            if len(splits) != 5:
                continue
            self._filenames.append(splits[0])
            self.y.append(int(splits[2]))
        self.x = self.ImageNetValDataX(dir, self._filenames, fashion, size, transform)
        self.y = np.array(self.y, dtype=int)

    def __len__(self):
        return len(self._filenames)

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
    return ['vgg16', 'vgg19', 'resnet50', 'resnet101',
            'resnet152', 'resnet50_v2', 'resnet101_v2',
            'resnet152_v2', 'mobilenet', 'mobilenet_v2',
            'inception_resnet_v2', 'inception_v3', 'xception',
            'densenet121', 'densenet169', 'densenet201',
            'nasnet_mobile', 'nasnet_large']


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
    model : instance of keras.Model
        Pretrained model to evaluate.
    data_normalized: instance of evaldnn.utils.keras.ImageNetValData
        Normalized data, used to do predictions and get intermediate outputs.
    data_original: instance of evaldnn.utils.keras.ImageNetValData
        Original data, used to perform adversarial attack.
    mean : tuple
        Mean of images.
    std : tuple
        Standard deviation of images.
    flip_axis : integer or None
        Indicate whether or not inputs should be flipped.
    bounds : tuple of length 2
        The bounds for the pixel values.

    """
    keras.backend.set_learning_phase(0)
    if model_name == 'vgg16':
        model = keras.applications.VGG16()
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x[..., ::-1] - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = -1
    elif model_name == 'vgg19':
        model = keras.applications.VGG19()
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x[..., ::-1] - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = -1
    elif model_name == 'resnet50':
        model = keras.applications.ResNet50()
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x[..., ::-1] - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = -1
    elif model_name == 'resnet101':
        model = keras.applications.ResNet101()
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x[..., ::-1] - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = -1
    elif model_name == 'resnet152':
        model = keras.applications.ResNet152()
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x[..., ::-1] - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = -1
    elif model_name == 'resnet50_v2':
        keras.applications.ResNet50V2()
        base_model = keras.applications.ResNet50V2(weights=None, include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(1000, activation='softmax', name='probs')(x)
        model = keras.Model(base_model.input, x)
        model.load_weights(common.user_home_dir() + '/.keras/models/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'resnet101_v2':
        keras.applications.ResNet101V2()
        base_model = keras.applications.ResNet101V2(weights=None, include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(1000, activation='softmax', name='probs')(x)
        model = keras.Model(base_model.input, x)
        model.load_weights(common.user_home_dir() + '/.keras/models/resnet101v2_weights_tf_dim_ordering_tf_kernels.h5')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'resnet152_v2':
        keras.applications.ResNet152V2()
        base_model = keras.applications.ResNet152V2(weights=None, include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(1000, activation='softmax', name='probs')(x)
        model = keras.Model(base_model.input, x)
        model.load_weights(common.user_home_dir() + '/.keras/models/resnet152v2_weights_tf_dim_ordering_tf_kernels.h5')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'mobilenet':
        model = keras.applications.MobileNet()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=224, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'mobilenet_v2':
        model = keras.applications.MobileNetV2()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=224, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'nasnet_mobile':
        model = keras.applications.NASNetMobile()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=224, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=224, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'nasnet_large':
        model = keras.applications.NASNetLarge()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=331, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=331, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'inception_resnet_v2':
        model = keras.applications.InceptionResNetV2()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'inception_v3':
        model = keras.applications.InceptionV3()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'xception':
        model = keras.applications.Xception()
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(fashion='inception_preprocessing', size=299, transform=lambda x: (x - mean) / std)
        data_original = ImageNetValData(fashion='inception_preprocessing', size=299, transform=None, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'densenet121':
        model = keras.applications.DenseNet121()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x / 255.0 - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: x / 255.0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'densenet169':
        model = keras.applications.DenseNet169()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x / 255.0 - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: x / 255.0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    elif model_name == 'densenet201':
        model = keras.applications.DenseNet201()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_normalized = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: (x / 255.0 - mean) / std)
        data_original = ImageNetValData(fashion='vgg_preprocessing', size=224, transform=lambda x: x / 255.0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
        bounds = (0, 255)
        flip_axis = None
    else:
        raise Exception('Invalid model name: ' + model_name + '. Available model names :' + str(imagenet_benchmark_zoo_model_names()))
    return model, data_normalized, data_original, mean, std, flip_axis, bounds


def cifar10_data(train=False, num_max=None):
    """ Load cifar10 data.

    Keras dataset utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    y_train = y_train.flatten().astype('int32')
    x_test = x_test.astype('float32') / 255
    y_test = y_test.flatten().astype('int32')
    if num_max is not None:
        x_train = x_train[:num_max]
        y_train = y_train[:num_max]
        x_test = x_test[:num_max]
        y_test = y_test[:num_max]
    if not train:
        return x_test, y_test
    else:
        return x_train, y_train


def mnist_data(train=False, num_max=None):
    """ Load mnist data.

    Keras dataset utils are used to download and load the data.

    Parameters
    ----------
    train : bool
        If it is false, test data will be loaded. Otherwise, train data will be
        used.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = y_train.flatten().astype('int32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_test = y_test.flatten().astype('int32')
    if num_max is not None:
        x_train = x_train[:num_max]
        y_train = y_train[:num_max]
        x_test = x_test[:num_max]
        y_test = y_test[:num_max]
    if not train:
        return x_test, y_test
    else:
        return x_train, y_train
