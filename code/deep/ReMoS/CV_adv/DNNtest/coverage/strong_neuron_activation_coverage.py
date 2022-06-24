"""
Provides a class for model neuron coverage evaluation.
"""

from __future__ import absolute_import

import numpy as np
from numba import njit, prange

from .utils import common
from .my_neuron_coverage import MyNeuronCoverage
from pdb import set_trace as st

class StrongNeuronActivationCoverage(MyNeuronCoverage):
    def __init__(self, k=5):
        super(StrongNeuronActivationCoverage, self).__init__(threshold=k)
        self._threshould = k
        self._topk = k
        assert isinstance(k, int)
        
    
    @staticmethod
    @njit(parallel=True)
    def _calc_1(intermediate_layer_output, features_index, k):
        """Calculate the mean of each output from each neuron in the layer and
        keep the maximum

        Parameters
        ----------
        intermediate_layer_output : array
            The scaled output of one intermediate layer.
        features_index : integer
            The index of feature in each intermediate layer output array.
            It should be either 0 or -1.

        Returns
        -------
        array
            The maximum output mean of each neuron.

        Notes
        -------
        This method is only used for intermediate output which has more than
        one dimension.
        This method is accelerated with numba. Link: http://numba.pydata.org/

        """
        num_layer_neuron = intermediate_layer_output[0].shape[features_index]
        num_input = len(intermediate_layer_output)
        
        result = np.zeros(shape=(num_input, num_layer_neuron), dtype=np.uint8)
        for input_id in prange(intermediate_layer_output.shape[0]):
            layer_neurons = []
            for layer_neuron_id in prange(num_layer_neuron):
                if features_index == -1:
                    neuron_output = intermediate_layer_output[input_id][..., layer_neuron_id]
                else:
                    neuron_output = intermediate_layer_output[input_id][layer_neuron_id]
                mean = np.mean(neuron_output)
                layer_neurons.append(mean)
            layer_neurons = np.array(layer_neurons)
            neuron_min = np.min(layer_neurons)
            neuron_max = np.max(layer_neurons)
            interval = (neuron_max - neuron_min) / k
            strong_thred = neuron_max - interval
            active_idxs = np.argwhere(layer_neurons > strong_thred)
            for neuron_idx in active_idxs:
                result[input_id][neuron_idx] = 1
        return result

    @staticmethod
    @njit(parallel=True)
    def _calc_2(intermediate_layer_output, features_index, threshold):
        """Calculate the mean of each output from each neuron in the layer and
        keep the maximum

        Parameters
        ----------
        intermediate_layer_output : array
           The scaled output of one intermediate layer.
        features_index : integer
           The index of feature in each intermediate layer output array.
           It should be either 0 or -1.

        Returns
        -------
        array
           The maximum output mean of each neuron.

        Notes
        -------
        This method is only used for intermediate output which has only one
        dimension.
        This method is accelerated with numba. Link: http://numba.pydata.org/

        """
        num_layer_neuron = intermediate_layer_output[0].shape[features_index]
        num_input = len(intermediate_layer_output)
        
        result = np.zeros(shape=(num_input, num_layer_neuron), dtype=np.uint8)
        for input_id in prange(intermediate_layer_output.shape[0]):
            layer_neurons = []
            for layer_neuron_id in prange(num_layer_neuron):
                if features_index == -1:
                    neuron_output = intermediate_layer_output[input_id][..., layer_neuron_id]
                else:
                    neuron_output = intermediate_layer_output[input_id][layer_neuron_id]
                layer_neurons.append(neuron_output)
            layer_neurons = np.array(layer_neurons)
            neuron_min = np.min(layer_neurons)
            neuron_max = np.max(layer_neurons)
            interval = (neuron_max - neuron_min) / k
            strong_thred = neuron_max - interval
            active_idxs = np.argwhere(layer_neurons > strong_thred)
            for neuron_idx in active_idxs:
                result[input_id][neuron_idx] = 1
        return result
