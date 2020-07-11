from lib.core.node.frame_reader import FrameReader


class TwoNumberSumNN(FrameReader):

    ''' Neural Network Framing Test for testing purposes. '''

    def __init__(self, number_one, number_two, node_type='SIGMOID'):
        self.number_of_hidden_layers = 2
        self.depth_of_hidden_layers = 5
        self.node_type = node_type
        self.number_one = number_one
        self.number_two = number_two
        super().__init__()
        self.desired_output = self.number_one + self.number_two

    def define_frame(self):
        ''' Test Neural Network. '''
        # [self.addLayer(self.depth_of_hidden_layers, self.node_type) for layer in range(self.number_of_hidden_layers)]
        for layer in range(self.number_of_hidden_layers):
            self.addLayer(self.depth_of_hidden_layers, self.node_type)
        inputs = [self.number_one, self.number_two]
        self.setInput(inputs)
        self.setOutputDepth(1)
