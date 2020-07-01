from math import tanh
from math import pow as power
from numpy import subtract

from lib.core.activation.activation import ActivationFunction


class TanhActivation(ActivationFunction):
    ''' Tanh Activation-Function Method Class. '''

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of Hyperbolic Tangent Activation Function '''
        val = input_array
        return tanh(val)

    def getDerivative(self):
        resultant = self.resultant
        return subtract(1, power(resultant, 2))
