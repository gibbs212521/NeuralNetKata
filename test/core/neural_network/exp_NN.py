from lib.core.neural_network.neural_network import NeuralNetwork


class ExponentNN(NeuralNetwork):

    ''' Neural Network Framing Test for testing purposes. '''

    def __init__(self, base_value, power_value, node_type='SIGMOID'):
        self.power_value = power_value
        self.number_of_hidden_layers = 2
        self.depth_of_hidden_layers = 5
        self.node_type = node_type
        self.base_value = base_value
        self.desired_output = self.base_value**self.power_value
        super().__init__()
        self.initial_error = self.desired_output - self.base_frame.layers['OUTPUT']

    def define_frame(self):
        ''' Test Neural Network. '''
        # [self.addLayer(self.depth_of_hidden_layers, self.node_type) for layer in range(self.number_of_hidden_layers)]
        for layer in range(self.number_of_hidden_layers):
            self.addLayer(self.depth_of_hidden_layers, self.node_type)
        self.setInput()
        self.setOutputDepth(1)

    def setInput(self, base_value=None, power_value=None):
        if base_value is None:
            base_value = self.base_value
        else:
            self.base_value = base_value
        if power_value is None:
            power_value = self.power_value
        else:
            self.power_value = power_value
        super().setInput([self.base_value, self.power_value])

    def runBackpropagation(self):
        super().runBackpropagation(self.error)

    def generalForwardPropagation(self, inputs):
        ''' General FIR Forward Propagation calculating overall error. '''
        total_inputs = len(inputs)
        general_error = 0
        for input_value in inputs:
            for exp_value in inputs:
                self.setInput(input_value, exp_value)
                self.forwardPropagation()
                general_error += self.error / total_inputs
        self.error = general_error

    def calculateError(self):
        self.desired_output = self.base_value**self.power_value
        self.error = self.desired_output - self.base_frame.layers['OUTPUT']
