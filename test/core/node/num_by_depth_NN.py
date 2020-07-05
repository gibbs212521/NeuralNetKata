from lib.core.node.frame_reader import FrameReader


class NumByDepthNN(FrameReader):

    ''' Neural Network Framing Test for testing purposes. '''

    def __init__(self, hid_layers_num, layer_depth, node_type='SIGMOID'):
        self.number_of_hidden_layers = hid_layers_num
        self.depth_of_hidden_layers = layer_depth
        self.node_type = node_type
        super().__init__()

    def define_frame(self):
        ''' Test Neural Network. '''
        # [self.addLayer(self.depth_of_hidden_layers, self.node_type) for layer in range(self.number_of_hidden_layers)]
        for layer in range(self.number_of_hidden_layers):
            self.addLayer(self.depth_of_hidden_layers, self.node_type)
