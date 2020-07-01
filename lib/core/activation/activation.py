from numpy import sum as arraySum


class ActivationFunction():
    ''' Base Activation-Function Method Class. '''

    __class_name__ = 'ActivationFunction'

    def __init__(self, reference_caller, input_array):
        self.caller = reference_caller

        self.input_sum = arraySum(input_array)
        self.resultant = self.getResultant(self.input_sum)

    def getResultant(self, inputArray=None):
        ''' Subclass Specific returning activation function results '''
        raise NotImplementedError('getResultant not defined', self.__class__)

    def getDerivative(self):
        ''' Subclass Specific returning activation function results '''
        raise NotImplementedError('getDerivative not defined', self.__class__)
