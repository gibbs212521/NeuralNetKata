from numpy.random import random
from numpy import multiply, divide, subtract


class WeightBiasFrame():
    ''' Data-Model Class Frame of Weights and Biases '''

    def __init__(self, base_frame):
        self.base_frame = base_frame
        self.layers = []
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
        for indx, layer in enumerate(self.base_frame):
            # TODO: Break into threaded operation
            layer_index = indx + 1
            if (layer_index) < len(self.base_frame):
                self.generateBiasWeightArray(layer, self.base_frame[layer_index])

    def getLayerWeightsAndBiases(self, layer_index=None):
        ''' Returns Weights and Biases Array for given layer_index. '''
        return self.layers[layer_index]