from lib.core.neural_network.neural_network import NeuralNetwork


class UnityOutputNN(NeuralNetwork):

    ''' Neural Network Framing Test for testing purposes. '''

    def __init__(self, input_set, node_type='SIGMOID'):
        self.number_of_hidden_layers = 2
        self.depth_of_hidden_layers = 5
        self.node_type = node_type
        self.input_set = input_set
        self.desired_output = input_set
        super().__init__()
        self.initial_error = self.desired_output - self.base_frame.layers['OUTPUT']

    def define_frame(self):
        ''' Test Neural Network. '''
        # [self.addLayer(self.depth_of_hidden_layers, self.node_type) for layer in range(self.number_of_hidden_layers)]
        for layer in range(self.number_of_hidden_layers):
            self.addLayer(self.depth_of_hidden_layers, self.node_type)
        self.setInput()
        self.setOutputDepth(len(self.input_set))

    def setInput(self, input_set=None):
        if input_set is None:
            input_set = self.input_set
        else:
            self.input_set = input_set
        super().setInput(self.input_set)

    def runBackpropagation(self):
        super().runBackpropagation(self.error)

    def calculateError(self):
        self.desired_output = self.input_set
        self.error = self.desired_output - self.base_frame.layers['OUTPUT']
