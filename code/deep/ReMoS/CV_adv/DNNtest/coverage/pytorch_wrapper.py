"""
Provides a class for torch model evaluation.
"""

from __future__ import absolute_import

import warnings

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

    # def intermediate_layer_outputs(self, dataset, callbacks, batch_size=8):
    #     """Get the intermediate layer outputs of the model.

    #     The method will use the model to do prediction batch by batch. For
    #     every batch, the the intermediate layer outputs will be captured and
    #     callback functions will be invoked. all intermediate layer output
    #     will be passed to the callback functions to do further process.

    #     Parameters
    #     ----------
    #     dataset : instance of torch.utils.data.Dataset
    #         Dataset from which to load the data.
    #     callbacks : list of functions
    #         Callback functions, each of which will be invoked when a batch is done.
    #     batch_size : integer
    #         Batch size for getting intermediate layer outputs.

    #     See Also
    #     --------
    #     :class:`metrics.neuron_coverage.NeuronCoverage`

    #     """
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    #     y_mini_batch_outputs = []
    #     hook_handles = []
    #     intermediate_layers = self._intermediate_layers(self._model)
    #     for intermediate_layer in intermediate_layers:
    #         def hook(module, input, output):
    #             y_mini_batch_outputs.append(output)

    #         handle = intermediate_layer.register_forward_hook(hook)
    #         hook_handles.append(handle)
    #     with torch.no_grad():
    #         for data in dataloader:
    #             if isinstance(data, list):
    #                 data = data[0]
    #             y_mini_batch_outputs.clear()
    #             data = data.to(self._device)
    #             self._model(data)
    #             for callback in callbacks:
    #                 callback(y_mini_batch_outputs, 0)
    #     for handle in hook_handles:
    #         handle.remove()
            
    def one_sample_intermediate_layer_outputs(self, sample, callbacks, ):
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        for intermediate_layer in intermediate_layers:
            def hook(module, input, output):
                y_mini_batch_outputs.append(output)
        
            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)
        
        with torch.no_grad():
            y_mini_batch_outputs.clear()
            output = self._model(sample)
            for callback in callbacks:
                callback(y_mini_batch_outputs, 0)
        for handle in hook_handles:
            handle.remove()
        
        return output
    
    def one_sample_intermediate_layer_outputs_with_grad(self, sample, callbacks, batch_size=8, return_intermedia_outputs=False):
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        
        for intermediate_layer in intermediate_layers:
            def hook(module, input, output):
                y_mini_batch_outputs.append(output)

            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)
        y_mini_batch_outputs.clear()
        output = self._model(sample)
        
        for callback in callbacks:
            callback(y_mini_batch_outputs, 0)
        for handle in hook_handles:
            handle.remove()
        
        compressed_y_mini_batch_outputs = []
        for item in y_mini_batch_outputs:
            if len(item.shape) == 4:
                mean = item.mean(-1).mean(-1)
            elif len(item.shape) == 2:
                mean = item
            else:
                raise NotImplementedError
            # print(mean.shape)
            compressed_y_mini_batch_outputs.append(mean)
        
        if return_intermedia_outputs:
            return output, compressed_y_mini_batch_outputs
        else:
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
        intermediate_layers = []
        for name, submodule in module.named_children():
            full_name = f"{pre_name}.{name}"
            if len(submodule._modules) > 0:
                intermediate_layers += self._intermediate_layers(submodule, full_name)
            else:
                # if 'Dropout' in str(submodule.type) or 'BatchNorm' in str(submodule.type) or 'ReLU' in str(submodule.type):
                if 'Dropout' in str(submodule.type) or 'ReLU' in str(submodule.type) or 'Linear' in str(submodule.type) or 'Pool' in str(submodule.type):
                        continue
                if self.intermedia_mode == "layer":
                    if type(self._model).__name__ == "ResNet":
                        if not full_name[-5:] == "1.bn2":
                            continue
                    
                else:
                    ...
                intermediate_layers.append(submodule)
                # print(full_name, )
        
        return intermediate_layers
