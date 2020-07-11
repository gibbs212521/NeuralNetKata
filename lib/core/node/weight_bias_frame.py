from numpy.random import random
from numpy import multiply, divide, subtract, ones, array


class WeightBiasFrame():
    ''' Data-Model Class Frame of Weights and Biases '''

    def __init__(self, base_frame):
        self.base_frame = base_frame
        self.frames = base_frame.frame
        self.layers = []
        self.delta_weights = []
        self.delta_biases = []
        self.layer_ranges = []  #### Later Feature to be implemented
        self.resetWeightBiasFrame()


    @staticmethod
    def normalRandomArray(rows, sub_rows, columns=2, layer_range=(-1, 1)):
        ''' Frequent method generating normal-random value nodes. '''
        # rows      ::  Current Layer Nodes
        # sub_rows  ::  Previous Layer Nodes
        # columns   ::  Weight & Bias
        #### NOTE: For increased processing speed, bias may be considered independent part of base frame nodes.
        ####       I, CG, have already committed to this less efficient architecture. FYI
        array = subtract(multiply(random((rows, sub_rows, columns)),\
            (layer_range[1]-layer_range[0])),\
            divide(layer_range[0] + layer_range[1], 2))
        return array

    def generateBiasWeightArray(self, prev_layer_length, current_layer_length):
        ''' Frequent method generating random weights and biases for current layer. '''
        self.layers.append(self.normalRandomArray(current_layer_length, prev_layer_length))

    def resetWeightBiasFrame(self):
        self.layers.clear()
        self.delta_weights.clear()
        self.delta_biases.clear()
        final_index = len(self.frames) - 1
        for indx, frames in enumerate(self.frames):
            # TODO: Break into threaded operation
            layer_index = indx + 1
            prev_layer = frames
            if layer_index <= final_index:
                current_layer = self.frames[indx+1]
                self.generateBiasWeightArray(prev_layer, current_layer)
                self.delta_weights.append(ones((current_layer, prev_layer)))
                self.delta_biases.append(ones((current_layer, prev_layer)))

    def getLayerWeightsAndBiases(self, layer_index=None):
        ''' Returns Weights and Biases Array for given layer_index. '''
        return self.layers[layer_index-1]

    def setLayerWeightsAndBiasesValues(self, input_nodes, layer_index):
        '''
        Change Node Values of given Layer to some numpy array.
        The second input, layer_title, may take layer_index as well.
        '''
        # TODO: Considering MultiThreading Method
        layer_title = self.base_frame.getLayerTitle(layer_index)
        self.layers[layer_index] = input_nodes

    def setWeightAndBiasValue(self, node_value, node_index, layer_index):
        ''' Change Value of Node in given Layer. '''
        self.layers[layer_index][node_index] = node_value

    def setDerivatives(self, layer_index, delta_weights, delta_biases, learning_rate=1):
        ''' Set Derivatives of Weights and Biases for specified layer. '''
        self.delta_weights[layer_index-1] = delta_weights * learning_rate
        self.delta_biases[layer_index-1] = delta_biases * learning_rate
