from numpy import ones, multiply as Numpy_Multiply

class ActivationFunction():
    ''' Base Activation-Function Method Class. '''

    __class_name__ = 'ActivationFunction'

    def __init__(self, input_array, reference_caller=None):
        self.caller = reference_caller
        self.input_array = input_array
        if isinstance(input_array, int):
            self.inputs_depth = 1
        else:
            self.inputs_depth = len(input_array)
        self.resultant = self.getResultant(input_array)

    def getResultant(self, inputArray=None):
        ''' Subclass Specific returning activation function results '''
        raise NotImplementedError('getResultant not defined', self.__class__)

    def getDerivative(self):
        ''' Subclass Specific returning activation function results '''
        self.derivatives = None
        raise NotImplementedError('getDerivative not defined', self.__class__)

    def getWeightDerivative(self, input_array=None):
        ''' Not True Derivative. Simplified for expedience. '''
        if input_array is None:
            input_array = self.input_array
        shape = input_array.shape
        derivative_array = ones(shape)
        for indx in range(shape[1]):
            derivative_array[:, indx] = self.derivatives
        return Numpy_Multiply(derivative_array, input_array)

    def getBiasDerivative(self, input_array=None):
        ''' Not True Derivative. Simplified for expedience. '''
        if input_array is None:
            input_array = self.input_array
        shape = input_array.shape
        derivative_array = ones(shape)
        for indx in range(shape[1]):
            derivative_array[:, indx] = self.derivatives
        return derivative_array
