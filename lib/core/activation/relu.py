from numpy import divide as Numpy_Divide
from numpy import abs

from lib.core.activation.activation import ActivationFunction


class ReluActivation(ActivationFunction):
    ''' ReLU Activation-Function Method Class. Rectified Linear Unit'''
    # NOTE: ReLU is recommended for deep Neural Networks

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of ReLU Activation Function. '''
        val = input_array
        return Numpy_Divide(val + abs(val), 2)

    def getDerivative(self):
        val = self.resultant
        return Numpy_Divide(val + abs(val), (2*abs(val)))
