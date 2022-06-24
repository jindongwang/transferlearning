"""
Provides a class for model neuron coverage evaluation.
"""

from __future__ import absolute_import

import numpy as np
from numba import njit, prange
import copy

from .utils import common
from pdb import set_trace as st

class MyNeuronCoverage:
    """ Class for model neuron coverage evaluation.

    Based on the outputs of the intermediate layers, update and report
    the model neuron coverage accordingly.

    Parameters
    ----------
    thresholds : list of floats
        The thresholds of neuron activation. Each of them will be used
        to calculate the model neuron coverage respectively.

    """

    def __init__(self, threshold=0.5):
        self._threshold = threshold
        self._layer_neuron_id_to_global_neuron_id = {}
        self._global_neuron_id_to_layer_neuron_id = {}
        self._results = {}
        self._num_layer = 0
        self._num_neuron = 0
        self._num_input = 0
        self._report_layers_and_neurons = True
        self.result = None
        self.structure_initialized = False
        
    def init_structure(self, intermediate_layer_outputs, features_index):
        # Initialize the information about networks
        current_global_neuron_id = 0
        for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            intermediate_layer_output_single_input = intermediate_layer_output[0]
            num_layer_neuron = intermediate_layer_output_single_input.shape[features_index]
            for layer_neuron_id in range(num_layer_neuron):
                self._layer_neuron_id_to_global_neuron_id[(layer_id, layer_neuron_id)] = current_global_neuron_id
                self._global_neuron_id_to_layer_neuron_id[current_global_neuron_id] = (layer_id, layer_neuron_id)
                current_global_neuron_id += 1
            self._num_layer += 1
            self._num_neuron += num_layer_neuron
            
            

    def update(self, intermediate_layer_outputs, features_index):
        """Update model neuron coverage accordingly.

        With each value in thresholds as the neuron activation threshold,
        the neuron coverage will be re-calculated and updated accordingly.

        Parameters
        ----------
        intermediate_layer_outputs : list of arrays
            The outputs of the intermediate layers.
        features_index : integer
            The index of feature in each intermediate layer output array.
            It should be either 0 or -1.

        Notes
        -------
        This method can be invoked for many times in one instance which means
        that once the outputs of the intermediate layers for a batch is got,
        this method can be invoked to update the status. The neuron coverage
        will be updated for every invocation.

        """
        

        # copy and convert intermediate layer outputs to numpy array
        intermediate_layer_outputs_new = []
        for intermediate_layer_output in intermediate_layer_outputs:
            intermediate_layer_output = common.to_numpy(intermediate_layer_output)
            intermediate_layer_outputs_new.append(intermediate_layer_output)
        intermediate_layer_outputs = intermediate_layer_outputs_new
        
        if not self.structure_initialized:
            self.init_structure(intermediate_layer_outputs, features_index)
            self.structure_initialized = True

        # get number of inputs
        num_input = len(intermediate_layer_outputs[0])
        self._num_input += num_input

        # scale the output of each layer
        for layer_id in range(len(intermediate_layer_outputs)):
            intermediate_layer_outputs[layer_id] = self._scale(intermediate_layer_outputs[layer_id])

        current_result = []
        # calculate and update the neuron coverage based on scaled outputs
        for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            if len(intermediate_layer_output.shape) > 2:
                result = self._calc_1(intermediate_layer_output, features_index, self._threshold)
            else:
                result = self._calc_2(intermediate_layer_output, features_index, self._threshold)
            current_result.append(result)
        
        self.per_layer_result = copy.deepcopy(current_result)
        current_result = np.concatenate(current_result, axis=1)
        self.result = current_result

    def report(self, *args):
        """Report model neuron coverage.

        The neuron coverage info will be reported with each value in thresholds
        as the neuron activation threshold. Reported info includes report time,
        number of layers, number of neurons, number in inputs, threshold, neuron
        coverage, number of neurons covered.

        """
        # if self._report_layers_and_neurons:
        #     self._report_layers_and_neurons = False
        #     print('[NeuronCoverage] Time:{:s}, Layers: {:d}, Neurons: {:d}'.format(common.readable_time_str(), self._num_layer, self._num_neuron))
        # for threshold in self._thresholds:
        #     print('[NeuronCoverage] Time:{:s}, Num: {:d}, Threshold: {:.6f}, Neuron Coverage: {:.6f}({:d}/{:d})'.format(common.readable_time_str(), self._num_input, threshold, self.get(threshold), len([v for v in self._results[threshold] if v]), self._num_neuron))

        num_input, num_neuron = self.result.shape
        for input_id in range(num_input):
            coverage = np.sum(self.result[input_id]) / num_neuron
            print(f"[NeuronCoverage] layers: {self._num_layer}, neurons: {self._num_neuron}, input_id {input_id}, coverage: {coverage:.4f}[{np.sum(self.result[input_id])}/{num_neuron}]")


    def get(self, threshold):
        """Get model neuron coverage.

        Parameters
        ----------
        threshold : float
            The neuron activation threshold.

        Returns
        -------
        float
            Model neuron coverage with parameter as the neuron activation threshold.

        Notes
        -------
        The parameter threshold must be one value in the list thresholds.

        """
        # return len([v for v in self._results[threshold] if v]) / self._num_neuron if self._num_neuron != 0 else 0
        if self.result is None:
            raise RunTimeError(f"Result is None!")
        return self.result

    @staticmethod
    @njit(parallel=True)
    def _scale(intermediate_layer_output):
        """For each input, scale the output of one intermediate layer
        by (x - min) / (max - min).

        Parameters
        ----------
        intermediate_layer_output : array
            The output of one intermediate layer.

        Returns
        -------
        array
            The scaled output of the intermediate layer.

        Notes
        -------
        This method is accelerated with numba. Link: http://numba.pydata.org/

        """
        for input_id in prange(intermediate_layer_output.shape[0]):
            intermediate_layer_output[input_id] = (intermediate_layer_output[input_id] - intermediate_layer_output[input_id].min()) / (intermediate_layer_output[input_id].max() - intermediate_layer_output[input_id].min())
        return intermediate_layer_output

    @staticmethod
    @njit(parallel=True)
    def _calc_1(intermediate_layer_output, features_index, threshold):
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
            for layer_neuron_id in prange(num_layer_neuron):
                if features_index == -1:
                    neuron_output = intermediate_layer_output[input_id][..., layer_neuron_id]
                else:
                    neuron_output = intermediate_layer_output[input_id][layer_neuron_id]
                mean = np.mean(neuron_output)
                if mean > threshold:
                    result[input_id][layer_neuron_id] = 1
                    # result[layer_neuron_id] = mean
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
            for layer_neuron_id in prange(num_layer_neuron):
                if features_index == -1:
                    neuron_output = intermediate_layer_output[input_id][..., layer_neuron_id]
                else:
                    neuron_output = intermediate_layer_output[input_id][layer_neuron_id]
                if neuron_output > threshold:
                    result[input_id][layer_neuron_id] = 1
        return result
