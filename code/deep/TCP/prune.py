import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
from IPython import embed

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
    _, conv = list(model.features._modules.items())[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset <  len(list(model.features._modules.items())):
        res = list(model.features._modules.items())[layer_index+offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1

    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups)
            #bias = conv.bias)

    new_conv.bias = torch.nn.Parameter(conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    if filter_index > new_weights.shape[0]:
        return model
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,)
                #bias = next_conv.bias)
        next_new_conv.bias = torch.nn.Parameter(next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
         features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
                    [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
         del model.features
         del conv

         model.features = features

    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index], \
                [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels

        new_linear_layer = \
             torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                 old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
             old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
             old_weights[:, (filter_index + 1) * params_per_input_channel :]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

