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
        total_inputs = len(inputs)**2
        general_error = 0
        for number_one in inputs:
            for number_two in inputs:
                self.setInput(number_one, number_two)
                self.forwardPropagation()
                general_error += self.error / total_inputs
        self.error = general_error

    def calculateError(self):
        ''' Neural Network Specific Cost Function '''
        raise NotImplementedError('calculateError Function is Neural Network Specific.')

