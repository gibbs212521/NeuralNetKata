class ActivationFunction():
    ''' Base Activation-Function Method Class. '''

    __class_name__ = 'ActivationFunction'

    def __init__(self, input_array, reference_caller=None):
        self.caller = reference_caller

        self.resultant = self.getResultant(input_array)

    def getResultant(self, inputArray=None):
        ''' Subclass Specific returning activation function results '''
        raise NotImplementedError('getResultant not defined', self.__class__)

    def getDerivative(self):
        ''' Subclass Specific returning activation function results '''
        raise NotImplementedError('getDerivative not defined', self.__class__)
