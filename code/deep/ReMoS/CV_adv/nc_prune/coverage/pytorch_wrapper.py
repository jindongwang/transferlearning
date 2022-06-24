"""
Provides a class for torch model evaluation.
"""

from __future__ import absolute_import

import warnings
from functools import partial

import torch

from .utils import common
from pdb import set_trace as st

class PyTorchModel:
    """ Class for torch model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    model : instance of torch.nn.Module
        torch model to evaluate.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, model, intermedia_mode=""):
        assert isinstance(model, torch.nn.Module)
        self._model = model
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.eval()
        self._model.to(self._device)
        self.intermedia_mode = intermedia_mode
        self.full_names = []
        self._intermediate_layers(self._model)

    def predict(self, dataset, callbacks, batch_size=16):
        """Predict with the model.

        The method will use the model to do prediction batch by batch. For
        every batch, callback functions will be invoked. Labels and predictions
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        dataset : instance of torch.utils.data.Dataset
            Dataset from which to load the data.
        callbacks : list of functions
            Callback functions, each of which will be invoked when a batch is done.
        batch_size : integer
            Batch size for prediction

        See Also
        --------
        :class:`metrics.accuracy.Accuracy`

        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self._device)
                labels = labels.to(self._device)
                y_mini_batch_pred = self._model(data)
                for callback in callbacks:
                    callback(labels, y_mini_batch_pred)

    def one_sample_intermediate_layer_outputs(self, sample, callbacks, ):
        y_mini_batch_outputs, y_mini_batch_inputs = {}, {}
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        for name, intermediate_layer in intermediate_layers.items():
            def hook(module, input, output, name):
                # y_mini_batch_outputs.append(output)
                # y_mini_batch_inputs.append(input[0])
                y_mini_batch_outputs[name] = output
                y_mini_batch_inputs[name] = input[0]
            
            hook_fn = partial(hook, name=name)
            handle = intermediate_layer.register_forward_hook(hook_fn)
            hook_handles.append(handle)
        
        with torch.no_grad():
            y_mini_batch_inputs.clear()
            y_mini_batch_outputs.clear()
            output = self._model(sample)
            for callback in callbacks:
                callback(y_mini_batch_inputs, y_mini_batch_outputs, 0)
        for handle in hook_handles:
            handle.remove()
        
        return output
    

    def _intermediate_layers(self, module, pre_name=""):
        """Get the intermediate layers of the model.

        The method will get some intermediate layers of the model which might
        be useful for neuron coverage computation. Some layers such as dropout
        layers are excluded empirically.

        Returns
        -------
        list of torch.nn.modules
            Intermediate layers of the model.

        """
        intermediate_layers = {}
        
        for name, submodule in module.named_children():
            if pre_name == "":
                full_name = f"{name}"
            else:
                full_name = f"{pre_name}.{name}"
            if len(submodule._modules) > 0:
                intermediate_layers.update(self._intermediate_layers(submodule, full_name))
            else:
                # if 'Dropout' in str(submodule.type) or 'BatchNorm' in str(submodule.type) or 'ReLU' in str(submodule.type):
                # if 'Dropout' in str(submodule.type) or 'ReLU' in str(submodule.type) or 'Linear' in str(submodule.type) or 'Pool' in str(submodule.type):
                #         continue
                if not "Conv2d" in str(submodule.type):
                    continue
                if self.intermedia_mode == "layer":
                    if type(self._model).__name__ == "ResNet":
                        if not full_name[-5:] == "1.bn2":
                            continue
                    
                else:
                    ...
                intermediate_layers[full_name] = submodule
                if full_name not in self.full_names:
                    
                    self.full_names.append(full_name)
                # print(full_name, )
        
        return intermediate_layers
