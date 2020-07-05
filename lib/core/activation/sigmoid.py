from numpy import exp
from numpy import divide as Numpy_Divide
from numpy import multiply as Numpy_Multiply

from lib.core.activation.activation import ActivationFunction


class SigmoidActivation(ActivationFunction):
    ''' Sigmoid Activation-Function Method Class. '''

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of Sigmoid Activation Function '''
        val = input_array
        return Numpy_Divide(1, (1+exp(-val)))

    def getDerivative(self):
        resultant = self.resultant
        return Numpy_Multiply(resultant, (1-resultant))
