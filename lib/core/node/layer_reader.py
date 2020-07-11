from copy import deepcopy
from numpy import multiply, sum as npSum

from lib.core.activation.sigmoid import SigmoidActivation
from lib.core.activation.tanh import TanhActivation
from lib.core.activation.relu import ReluActivation
from lib.core.activation.leaky_relu import LeakyReluActivation
from lib.core.activation.node_sum import SumActivation

class LayerReader():

    '''
    Method Class for reading a layer. Accepts either layer_index or proper layer_names
    for layer_title input.
    '''

    valid_node_types = ['INPUT', 'SIMPLE_SUM', 'SIGMOID', 'TANH',\
        'RELU', 'LEAKY_RELU']
    node_dictionary = {}
    node_dictionary['INPUT'] = deepcopy
    node_dictionary['SIMPLE_SUM'] = SumActivation
    node_dictionary['SIGMOID'] = SigmoidActivation
    node_dictionary['TANH'] = TanhActivation
    node_dictionary['RELU'] = ReluActivation
    node_dictionary['LEAKY_RELU'] = LeakyReluActivation
    node_class = None

    def __init__(self, base_frame, weight_bias_frame, layer_title=None, learning_rate=0.005):
        self.learning_rate = learning_rate
        layer_index, current_nodes = self.calculateLayerForwardPropagation(base_frame,\
            weight_bias_frame, layer_title)
        base_frame.setLayerNodeValues(current_nodes, layer_index)

    def setNodeClass(self, node_type):
        ''' Sets Node Type for Layer being read. '''
        node_type = node_type.upper()
        self.node_class = self.node_dictionary[node_type]

    def calculateLayerForwardPropagation(self, base_frame, weight_bias_frame, layer_title):
        ''' Returns new node values for given node layer. '''
        # TODO: MultiThread Following Section:
        layer_title = base_frame.getLayerTitle(layer_title)
        layer_index, input_nodes, current_nodes, layer_type = base_frame.selectLayer(layer_title)
        self.setNodeClass(layer_type)
        weight_biases = weight_bias_frame.getLayerWeightsAndBiases(layer_index)
        if layer_type == 'INPUT':
            return layer_index, current_nodes
        node_inputs = deepcopy(weight_biases)
        node_inputs[:, :, 0] = multiply(weight_biases[:, :, 0], input_nodes)
        node_input_sums = npSum(npSum(node_inputs, 2), 1)
        layer_activation = self.node_class(node_input_sums)
        current_nodes = layer_activation.resultant
        delta_nodes = layer_activation.getDerivative()
        base_frame.setDerivatives(layer_title, delta_nodes, self.learning_rate))
        delta_weights = layer_activation.getWeightDerivative(weight_biases[:, :, 0])
        delta_biases = layer_activation.getBiasDerivative(weight_biases[:, :, 1])
        weight_bias_frame.setDerivatives(layer_index, delta_weights, delta_biases, self.learning_rate)
        return layer_index, current_nodes
