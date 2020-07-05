from numpy import sum as npSum

from lib.core.activation.activation import ActivationFunction


class SumActivation(ActivationFunction):
    ''' Sum Activation-Function Method Class. '''

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of Sigmoid Activation Function '''
        return npSum(input_array)

    @staticmethod
    def getDerivative(self):
        '''
        Derivative of Sum is ZERO.
        Layer Proportion should also be ZERO.
        '''
        return 0
