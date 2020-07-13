from lib.core.node.frame_reader import FrameReader

class NeuralNetwork(FrameReader):

    def trainNN(self, iterations=1, learning_rate=None):
        if learning_rate is None:
            pass
        else:
            self.learning_rate = float(learning_rate)
        for iteration in range(iterations):
            self.forwardPropagation()
            self.runBackpropagation()

    def trainFIRNeuralNetwork(self, inputs, iterations=1, learning_rate=None):
        if learning_rate is None:
            pass
        else:
            self.learning_rate = float(learning_rate)
        for iteration in range(iterations):
            self.generalForwardPropagation(inputs)
            self.runBackpropagation()

    def generalForwardPropagation(self, inputs):
        ''' General FIR Forward Propagation calculating overall error. '''
        total_inputs = len(inputs)
        general_error = 0
        for input_value in inputs:
            self.setInput(input_value)
            self.forwardPropagation()
            general_error += self.error / total_inputs
        self.error = general_error

    def calculateError(self):
        ''' Neural Network Specific Cost Function '''
        raise NotImplementedError('calculateError Function is Neural Network Specific.')
