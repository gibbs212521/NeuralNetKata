from numpy import divide as Numpy_Divide
from numpy import multiply as Numpy_Multiply

from lib.core.activation.activation import ActivationFunction


class LeakyReluActivation(ActivationFunction):
    ''' Leaky ReLU Activation-Function Method Class. '''

    def getResultant(self, input_array, ratio=0.075):
        ''' Get Results of Leaky ReLU Activation Function '''
        val = input_array
        self.negative_ratio = abs(ratio)
        return Numpy_Divide(abs(val) + val, 2) + Numpy_Multiply(Numpy_Divide(val - abs(val), 2), ratio)

    def getDerivative(self):
        val = self.input_sum
        return Numpy_Divide(abs(val) + val, (2*abs(val))) + Numpy_Multiply(Numpy_Divide(abs(val) - val, (2*abs(val))), self.negative_ratio)
