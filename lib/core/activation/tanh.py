from numpy import subtract, tanh, power, ones,\
    multiply as Numpy_Multiply

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
        self.derivatives = subtract(1, power(resultant, 2))
        return self.derivatives
