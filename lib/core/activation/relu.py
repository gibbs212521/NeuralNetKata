from numpy import divide as Numpy_Divide

from lib.core.activation.activation import ActivationFunction


class ReluActivation(ActivationFunction):
    ''' ReLU Activation-Function Method Class. '''

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of ReLU Activation Function '''
        val = input_array
        return Numpy_Divide(val + abs(val), 2)

    def getDerivative(self):
        val = self.input_sum
        return Numpy_Divide(val + abs(val), (2*abs(val)))
