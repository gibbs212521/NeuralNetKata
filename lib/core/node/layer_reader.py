from copy import deepcopy
from numpy import multiply, sum as npSum

from lib.core.activation.sigmoid import SigmoidActivation
from lib.core.activation.tanh import TanhActivation
from lib.core.activation.relu import ReluActivation
from lib.core.activation.leaky_relu import LeakyReluActivation

class LayerReader():

    '''
    Method Class for reading a layer. Accepts either layer_index or proper layer_names
    for layer_title input.
    '''

    valid_node_types = ['INPUT', 'SIMPLE_SUM', 'SIGMOID', 'TANH',\
        'RELU', 'LEAKY_RELU']
    node_dictionary = {}
    node_dictionary['INPUT'] = deepcopy
    node_dictionary['SIMPLE_SUM'] = npSum
    node_dictionary['SIGMOID'] = SigmoidActivation
    node_dictionary['TANH'] = TanhActivation
    node_dictionary['RELU'] = ReluActivation
    node_dictionary['LEAKY_RELU'] = LeakyReluActivation
    node_class = None

    def __init__(self, base_frame, weight_bias_frame, layer_title=None):
        layer_index, current_nodes = self.calculateLayerForwardPropagation(base_frame,\
            weight_bias_frame, layer_title)
        base_frame.setLayerNodeValues(current_nodes, layer_index)

    def resetNodeClass(self, node_type):
        ''' Sets Node Type for Layer being read. '''
        node_type = node_type.upper()
        self.node_class = self.node_dictionary[node_type]

    def calculateLayerForwardPropagation(self, base_frame, weight_bias_frame, layer_title):
        ''' Returns new node values for given node layer. '''
        # TODO: MultiThread Following Section:
        layer_index, input_nodes, current_nodes, layer_type = base_frame.selectLayer(layer_title)
        self.resetNodeClass(layer_type)
        weight_biases = weight_bias_frame.getLayerWeightsAndBiases(layer_index)
        node_input_sums = deepcopy(weight_biases)
        node_input_sums[:, :, 0] = multiply(weight_biases[:, :, 0], input_nodes)
        node_input_sums = npSum(npSum(node_input_sums, 2), 1)
        current_nodes = self.node_class(node_input_sums)
        return layer_index, current_nodes
