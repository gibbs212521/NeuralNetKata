from lib.core.neural_network.neural_network import NeuralNetwork


class TwoNumberSumNN(NeuralNetwork):

    ''' Neural Network Framing Test for testing purposes. '''

    def __init__(self, number_one, number_two, node_type='SIGMOID'):
        self.number_of_hidden_layers = 2
        self.depth_of_hidden_layers = 5
        self.node_type = node_type
        self.number_one = number_one
        self.number_two = number_two
        self.desired_output = self.number_one + self.number_two
        super().__init__()
        self.initial_error = self.desired_output - self.base_frame.layers['OUTPUT']

    def define_frame(self):
        ''' Test Neural Network. '''
        # [self.addLayer(self.depth_of_hidden_layers, self.node_type) for layer in range(self.number_of_hidden_layers)]
        for layer in range(self.number_of_hidden_layers):
            self.addLayer(self.depth_of_hidden_layers, self.node_type)
        self.setInput(self.number_one, self.number_two)
        self.setOutputDepth(1)

    def setInput(self, number_one, number_two):
        self.number_one = number_one
        self.number_two = number_two
        super().setInput([number_one, number_two])

    def generalForwardPropagation(self, inputs):
        total_inputs = len(inputs)**2
        general_error = 0
        for number_one in inputs:
            for number_two in inputs:
                self.setInput(number_one, number_two)
                self.forwardPropagation()
                general_error += self.error / total_inputs
        self.error = general_error

    def runBackpropagation(self):
        super().runBackpropagation(self.error)

    def calculateError(self):
        self.desired_output = self.number_one + self.number_two
        self.error = self.desired_output - self.base_frame.layers['OUTPUT']
