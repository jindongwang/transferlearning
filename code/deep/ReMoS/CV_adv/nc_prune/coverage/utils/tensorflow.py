"""
Provides some useful utils for tensorflow model evaluation.
"""

from __future__ import absolute_import

import random
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import nasnet
from tensorflow.contrib.slim.nets import pnasnet
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import vgg

from evaldnn.utils import common


class ImageNetValData():
    """ Class for loading and preprocessing imagenet validation set.

    One can download the imagenet validation set at http://image-net.org/.
    To use this class, one should also download ILSVRC2012_validation_ground_truth.txt
    and put it in the same directory as the imagenet validation set.

    Parameters
    ----------
    width : integer
        Target image width.
    height : integer
        Target image height.
    fashion : str
        Indicate the preprocessing fashion. It can be either vgg_preprocessing
        or inception_preprocessing.
    transform : function(image) -> image
        The transform function for preprocessing image after the image is loaded
        and cropped to proper size.
    label_offset : integer
        The offset of the label. For some models the offset should be set to 1
        because these models are trained with 1001 classes.
    shuffle : bool
        Indicate whether or not to shuffle the images.
    seed : integer
        Random seed used for shuffle.
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    """

    class ImageNetValDataX():

        def __init__(self, dir, filenames, width, height, fashion, transform):
            self._dir = dir
            self._filenames = filenames
            self._width = width
            self._height = height
            self._fashion = fashion
            self._transform = transform

        def __len__(self):
            return len(self._filenames)

        def __getitem__(self, index):
            tf.compat.v1.enable_eager_execution()
            x = None
            for filename in self._filenames[index]:
                path = self._dir + '/' + filename
                image = tf.image.decode_image(tf.io.read_file(path), channels=3)
                if self._fashion == 'vgg_preprocessing':
                    image = self._aspect_preserving_resize(image, 256)
                    image = self._central_crop([image], self._height, self._width)[0]
                    image.set_shape([self._height, self._width, 3])
                    image = tf.cast(image, dtype=tf.float32)
                elif self._fashion == 'inception_preprocessing':
                    image = tf.cast(image, tf.float32)
                    image.set_shape([tf.compat.v1.Dimension(None), tf.compat.v1.Dimension(None), tf.compat.v1.Dimension(3)])
                    image = tf.image.central_crop(image, central_fraction=0.875)
                    image = tf.expand_dims(image, 0)
                    image = tf.compat.v1.image.resize_bilinear(image, [self._width, self._height], align_corners=False)
                    image = tf.squeeze(image, [0])
                else:
                    raise Exception('Invalid fashion', self._fashion)
                if self._transform is not None:
                    image = self._transform(image)
                image = tf.expand_dims(image, axis=0)
                if x is None:
                    x = image
                else:
                    x = tf.concat([x, image], 0)
            x = x.numpy()
            tf.compat.v1.disable_eager_execution()
            return x

        def _smallest_size_at_least(self, height, width, smallest_side):
            smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
            height = tf.cast(height, dtype=tf.float32)
            width = tf.cast(width, dtype=tf.float32)
            smallest_side = tf.cast(smallest_side, dtype=tf.float32)
            scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
            new_height = tf.cast(tf.math.rint(height * scale), dtype=tf.int32)
            new_width = tf.cast(tf.math.rint(width * scale), dtype=tf.int32)
            return new_height, new_width

        def _aspect_preserving_resize(self, image, smallest_side):
            smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
            shape = tf.shape(image)
            height = shape[0]
            width = shape[1]
            new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
            image = tf.expand_dims(image, 0)
            resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
            resized_image = tf.squeeze(resized_image)
            resized_image.set_shape([None, None, 3])
            return resized_image

        def _central_crop(self, image_list, crop_height, crop_width):
            outputs = []
            for image in image_list:
                image_height = tf.shape(image)[0]
                image_width = tf.shape(image)[1]
                offset_height = (image_height - crop_height) / 2
                offset_width = (image_width - crop_width) / 2
                outputs.append(self._crop(image, offset_height, offset_width, crop_height, crop_width))
            return outputs

        def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
            original_shape = tf.shape(image)
            rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
            with tf.control_dependencies([rank_assertion]):
                cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
            size_assertion = tf.Assert(
                tf.logical_and(tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_width)), ['Crop size greater than the image size.'])
            offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), dtype=tf.int32)
            with tf.control_dependencies([size_assertion]):
                image = tf.slice(image, offsets, cropped_shape)
            return tf.reshape(image, cropped_shape)

    def __init__(self, width, height, fashion, transform=None, label_offset=0, shuffle=False, seed=None, num_max=None):
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
        self.y = []
        for line in lines:
            splits = line.split('---')
            if len(splits) != 5:
                continue
            self._filenames.append(splits[0])
            self.y.append(int(splits[2]))
        self.x = self.ImageNetValDataX(self._dir, self._filenames, width, height, fashion, transform)
        self.y = np.array(self.y, dtype=int) + label_offset

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
    return ['vgg16', 'vgg19', 'resnet_v1_101', 'resnet_v1_152',
            'resnet_v1_50', 'inception_v1', 'inception_v2',
            'inception_v3', 'inception_v4', 'inception_resnet_v2',
            'mobilenet_v1_0_5_160', 'mobilenet_v1_0_25_128',
            'mobilenet_v1_1_0_224', 'mobilenet_v2_1_0_224',
            'mobilenet_v2_1_4_224', 'resnet_v2_101',
            'resnet_v2_152', 'resnet_v2_50',
            'nasnet_a_mobile_224', 'pnasnet_5_mobile_224',
            'nasnet_a_large_331', 'pnasnet_5_large_331']


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
    session : `tensorflow.session`
        The session with which the graph will be computed.
    logits : `tensorflow.Tensor`
        The predictions of the model.
    inputs : `tensorflow.Tensor`
        The input to the model, usually a `tensorflow.placeholder`.
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
    tf.get_logger().setLevel('ERROR')
    if model_name == 'vgg16':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            logits, _ = vgg.vgg_16(input, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/vgg_16.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(224, 224, 'vgg_preprocessing', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = ImageNetValData(224, 224, 'vgg_preprocessing', transform=None, label_offset=0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'vgg19':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            logits, _ = vgg.vgg_19(input, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/vgg_19.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(224, 224, 'vgg_preprocessing', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = ImageNetValData(224, 224, 'vgg_preprocessing', transform=None, label_offset=0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v1_101':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                resnet_v1.resnet_v1_101(input, num_classes=1000, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v1_101/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v1_101.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(224, 224, 'vgg_preprocessing', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = ImageNetValData(224, 224, 'vgg_preprocessing', transform=None, label_offset=0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v1_152':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                resnet_v1.resnet_v1_152(input, num_classes=1000, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v1_152/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v1_152.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(224, 224, 'vgg_preprocessing', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = ImageNetValData(224, 224, 'vgg_preprocessing', transform=None, label_offset=0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v1_50':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                resnet_v1.resnet_v1_50(input, num_classes=1000, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v1_50/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v1_50.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_normalized = ImageNetValData(224, 224, 'vgg_preprocessing', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = ImageNetValData(224, 224, 'vgg_preprocessing', transform=None, label_offset=0, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v1':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/inception_v1.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v2':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(inception.inception_v2_arg_scope()):
                logits, _ = inception.inception_v2(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/inception_v2.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v3':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, _ = inception.inception_v3(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/inception_v3.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_v4':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(inception.inception_v4_arg_scope()):
                logits, _ = inception.inception_v4(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/inception_v4.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'inception_resnet_v2':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
                logits, _ = inception.inception_resnet_v2(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/inception_resnet_v2_2016_08_30.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v1_0_5_160':
        graph = tf.Graph()
        with tf.io.gfile.GFile(common.user_home_dir() + '/EvalDNN-models/tensorflow/mobilenet_v1_0.5_160/mobilenet_v1_0.5_160_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 160, 160, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV1/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(160, 160, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(160, 160, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v1_0_25_128':
        graph = tf.Graph()
        with tf.io.gfile.GFile(common.user_home_dir() + '/EvalDNN-models/tensorflow/mobilenet_v1_0.25_128/mobilenet_v1_0.25_128_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 128, 128, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV1/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(128, 128, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(128, 128, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v1_1_0_224':
        graph = tf.Graph()
        with tf.io.gfile.GFile(common.user_home_dir() + '/EvalDNN-models/tensorflow/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 224, 224, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV1/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v2_1_0_224':
        graph = tf.Graph()
        with tf.io.gfile.GFile(common.user_home_dir() + '/EvalDNN-models/tensorflow/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 224, 224, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV2/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'mobilenet_v2_1_4_224':
        graph = tf.Graph()
        with tf.io.gfile.GFile(common.user_home_dir() + '/EvalDNN-models/tensorflow/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 224, 224, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV2/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'nasnet_a_large_331':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 331, 331, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                logits, end_points = nasnet.build_nasnet_large(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/nasnet-a_large_04_10_2017/model.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(331, 331, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(331, 331, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'nasnet_a_mobile_224':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
                logits, end_points = nasnet.build_nasnet_mobile(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/nasnet-a_mobile_04_10_2017/model.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'pnasnet_5_large_331':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 331, 331, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(pnasnet.pnasnet_large_arg_scope()):
                logits, end_points = pnasnet.build_pnasnet_large(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/pnasnet-5_large_2017_12_13/model.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(331, 331, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(331, 331, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'pnasnet_5_mobile_224':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(pnasnet.pnasnet_mobile_arg_scope()):
                logits, end_points = pnasnet.build_pnasnet_mobile(input, 1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/pnasnet-5_mobile_2017_12_13/model.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(224, 224, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(224, 224, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v2_101':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
                resnet_v2.resnet_v2_101(input, num_classes=1001, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v2_101/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v2_152':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
                resnet_v2.resnet_v2_152(input, num_classes=1001, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v2_152/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    elif model_name == 'resnet_v2_50':
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
                resnet_v2.resnet_v2_50(input, num_classes=1001, is_training=False)
                logits = session.graph.get_tensor_by_name('resnet_v2_50/predictions/Reshape:0')
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, common.user_home_dir() + '/EvalDNN-models/tensorflow/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_normalized = ImageNetValData(299, 299, 'inception_preprocessing', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = ImageNetValData(299, 299, 'inception_preprocessing', transform=None, label_offset=1, shuffle=data_original_shuffle, seed=data_original_seed, num_max=data_original_num_max)
    else:
        raise Exception('Invalid model name: ' + model_name + '. Available model names :' + str(imagenet_benchmark_zoo_model_names()))
    bounds = (0, 255)
    return session, logits, input, data_normalized, data_original, mean, std, bounds


def cifar10_test_data(num_max=None):
    """ Load cifar10 data.

    Parameters
    ----------
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    Notes
    ----------
    Corresponding data should be downloaded manually in advance.

    """
    x_test = np.load(common.user_home_dir() + '/EvalDNN-data/cifar-10-tensorflow/x_test.npy')
    y_test = np.load(common.user_home_dir() + '/EvalDNN-data/cifar-10-tensorflow/y_test.npy')
    if num_max is not None:
        x_test = x_test[:num_max]
        y_test = y_test[:num_max]
    return x_test, y_test


def mnist_test_data(num_max=None):
    """ Load mnist data.

    Parameters
    ----------
    num_max : integer
        The maximum number of images to load. If it is set to none, all images
        will be loaded.

    Notes
    ----------
    Corresponding data should be downloaded manually in advance.

    """
    x_test = np.load(common.user_home_dir() + '/EvalDNN-data/MNIST/tensorflow/x_test.npy')
    y_test = np.load(common.user_home_dir() + '/EvalDNN-data/MNIST/tensorflow/y_test.npy')
    if num_max is not None:
        x_test = x_test[:num_max]
        y_test = y_test[:num_max]
    return x_test, y_test
