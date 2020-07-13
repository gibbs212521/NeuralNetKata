from numpy import array, subtract, multiply, ndarray, average,\
    sum as npSum

from lib.core.node.layer_reader import LayerReader
from lib.core.node.base_frame import BaseFrame
from lib.core.node.weight_bias_frame import WeightBiasFrame

class FrameReader():
    '''
    Abstract-Controller / Quasi-Method Class for handling and generating
    Neural Network Dataframes.
    '''

    valid_node_types = ['INPUT', 'SIMPLE_SUM', 'SIGMOID', 'TANH',\
        'RELU', 'LEAKY_RELU']

    base_frame_class = BaseFrame
    weight_bias_frame_class = WeightBiasFrame
    layer_reader_class = LayerReader
    output_type = 'SIMPLE_SUM'
    learning_rate = 0.1

    def __init__(self):

        self.node_depth_list = [1, 1]
        self.node_type_list = ['INPUT', 'SIMPLE_SUM'] # SIMPLE_SUM for output
        self.output_depth = 1
        self.input_depth = 1

        self.base_frame = self.base_frame_class()
        for node_type in self.valid_node_types:
            self.base_frame.addValidNodeType(node_type)
        self.define_frame()
        self.build_frame()
        self.weight_base_frame = self.weight_bias_frame_class(self.base_frame)
        self.forwardPropagation()

    def define_frame(self):
        '''
        Define Frame via 2 lists: node_depth_list & node_type_list.
        Do this either by direct instantiation or addLayer Method.
        '''
        raise NotImplementedError('Frame Reader Class requires subclass instantiation of Define Frame Method')

    def build_frame(self):
        ''' Builds out base_frame based on node_depth_list and node_type_list'''
        depth_length = len(self.node_depth_list)
        type_length = len(self.node_type_list)
        if depth_length != type_length:
            raise IndexError('The node_depth must be defined for each layer with node_type.')
        # self.base_frame.inputResetLayer(self.input_depth)
        self.base_frame.outputResetLayer(self.output_depth, self.output_type)
        for indx in range(depth_length-2):
            node_indx = indx + 1
            node_count = self.node_depth_list[node_indx]
            node_type = self.node_type_list[node_indx]
            # TODO: Implement UInt8 dtype for image processing.
            self.base_frame.addHiddenLayer(node_count, node_type)

    def addLayer(self, layer_depth, node_type):
        ''' Adds layer to base_frame when placed in subclass define_frame method. '''
        node_type = str(node_type).upper()
        self.base_frame.nodeTypeCheck(node_type)
        self.node_depth_list.insert(-1, int(layer_depth))
        self.node_type_list.insert(-1, node_type)

    def setInput(self, input_array):
        if isinstance(input_array, list) or isinstance(input_array, ndarray):
            self.input_depth = len(input_array)
            self.base_frame.inputResetLayer(self.input_depth)
            if isinstance(input_array, list):
                self.base_frame.layers['INPUT'] = array(input_array)
            else:
                self.base_frame.layers['INPUT'] = input_array
        else:
            raise TypeError('setInput input_array must be either a list or numpy array')
        self.base_frame.frame[0] = self.input_depth

    def setOutputDepth(self, depth=None):
        if depth is None:
            depth = self.output_depth
        if isinstance(depth, list) or isinstance(depth, ndarray):
            depth = len(depth)
            self.output_depth = depth
        self.base_frame.frame[-1] = depth

    def forwardPropagation(self):
        ''' Propagates values for base_frame and weight_bias_frame. '''
        for indx in range(len(self.node_type_list)):
            if indx == 0:
                continue
            self.layer_reader_class(self.base_frame, self.weight_base_frame, indx, self.learning_rate)
        self.calculateError()

    def runBackpropagation(self, net_output_error=1):
        ''' Backpropagate values from base_frame and weight_bias_frame. '''
        if isinstance(net_output_error, list) or isinstance(net_output_error, ndarray):
            net_output_error = average(net_output_error)
        shapes = [layer.shape for layer in self.weight_base_frame.layers]
        shapes.reverse()
        for indx in range(len(shapes)):
            layer_index = -(indx + 1)
            delta_biases = self.weight_base_frame.delta_biases[layer_index]
            delta_weights = self.weight_base_frame.delta_weights[layer_index]
            base_layer_index = self.base_frame.getLayerTitle(layer_index)
            # Derivatives compound due to Calculus Chain Rule as one moves towards inputs
            recursion_indx = layer_index + 1
            if recursion_indx is 0:
                continue
            node_count = len(self.base_frame.layers_delta[base_layer_index])
            layer_shape = delta_biases.shape
            recursive_delta_weights = self.weight_base_frame.delta_weights[recursion_indx]
            self.base_frame.layers_delta[base_layer_index] = [npSum(recursive_delta_weights[:, k]) for k in range(node_count)]
            delta_nodes = self.base_frame.layers_delta[base_layer_index]
            for layer_shape_indx in range(layer_shape[1]):
                self.weight_base_frame.delta_weights[layer_index][:, layer_shape_indx] = multiply(delta_weights[:, layer_shape_indx], delta_nodes)
                self.weight_base_frame.delta_biases[layer_index][:, layer_shape_indx] = multiply(delta_biases[:, layer_shape_indx], delta_nodes)
        weight_difference = multiply(array(self.weight_base_frame.delta_weights, dtype='object'), -net_output_error)
        bias_difference = multiply(array(self.weight_base_frame.delta_biases, dtype='object'), -net_output_error)
        # Brute forcing for time's sake *** NOT EFFICIENT
        dweights = array([item[:, :, 0] for item in self.weight_base_frame.layers], dtype='object')
        dweights = subtract(dweights, weight_difference)
        dbiases = array([item[:, :, 1] for item in self.weight_base_frame.layers], dtype='object')
        dbiases = subtract(dweights, bias_difference)
        for array_number, layer in enumerate(self.weight_base_frame.layers):
            for layer_number, nodes in enumerate(layer):
                for node_number in range(len(nodes)):
                    self.weight_base_frame.layers[array_number][layer_number][node_number][0] = dweights[array_number][layer_number][node_number]
                    self.weight_base_frame.layers[array_number][layer_number][node_number][1] = dbiases[array_number][layer_number][node_number]
