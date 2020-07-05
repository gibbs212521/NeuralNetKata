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
        self.base_frame.inputResetLayer(self.input_depth)
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

    def forwardPropagation(self):
        ''' Propagates values for base_frame and weight_bias_frame. '''
        for indx in range(len(self.node_type_list)):
            if indx == 0:
                continue
            self.layer_reader_class(self.base_frame, self.weight_base_frame, indx)
