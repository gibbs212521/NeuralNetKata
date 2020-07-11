from numpy import ones, array, ndarray, sum as npSum

from lib.core.activation.activation import ActivationFunction


class SumActivation(ActivationFunction):
    ''' Sum Activation-Function Method Class. '''

    @staticmethod
    def getResultant(input_array):
        ''' Get Results of Sigmoid Activation Function '''
        return npSum(input_array)

    @staticmethod
    def getDerivative():
        '''
        Derivative of Sum is ZERO.
        Layer Proportion should also be ZERO.
        '''
        return 0

    def getWeightDerivative(self, input_array=None):
        ''' Not True Derivative. Simplified for expedience. '''
        if input_array is None:
            input_array = self.inputs_depth
        elif isinstance(input_array, list):
            input_array = array(input_array)
        return input_array

    def getBiasDerivative(self, input_array=None):
        ''' Not True Derivative. Simplified for expedience. '''
        if input_array is None:
            shape = self.inputs_depth
        elif isinstance(input_array, list):
            shape = len(input_array)
        elif isinstance(input_array, ndarray):
            shape = input_array.shape
        return ones(shape)
