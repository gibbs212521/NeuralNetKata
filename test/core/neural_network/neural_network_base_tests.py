import unittest
from math import isnan

from test.core.neural_network.two_number_sum_NN import TwoNumberSumNN

class NeuralNetworkTestSuite(unittest.TestCase):

    ''' Unit Tests for Neural Network Method Classes '''

    def test_00_two_number_sum_overlearn_test(self):
        '''
        Elementary Neural Network Overfitting Test.
        Final Error should be less than 0.01% of initial.
        '''
        number_one = 1
        number_two = 2
        desired_number = number_one + number_two
        NNTest = TwoNumberSumNN(number_one, number_two)
        initial_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        NNTest.trainNN(10)
        NNTest.trainNN(100, learning_rate=0.05)
        NNTest.trainNN(100, learning_rate=0.001)
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        self.assertLess(final_error**2, initial_error**2/10000)
        # Demonstrate overfitting by handling value outside of scope.
        number_one = 500
        number_two = 1000
        desired_number = number_one + number_two
        NNTest.setInput(number_one, number_two)
        NNTest.forwardPropagation()
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        self.assertGreater(final_error, initial_error**2)

    def test_01_two_number_sum_fitness_test(self):
        inputs = range(-5, 5)
        number_one = 1
        number_two = 2
        desired_number = number_one + number_two
        NNTest = TwoNumberSumNN(number_one, number_two)
        initial_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        NNTest.trainFIRNeuralNetwork(inputs, 2, learning_rate=0.5)
        NNTest.trainFIRNeuralNetwork(inputs, 10, learning_rate=0.1)
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        self.assertLess(final_error**2, initial_error**2/10000)
        print(final_error)
        # Demonstrate overfitting by handling value outside of scope.
        number_one = 500
        number_two = 1000
        desired_number = number_one + number_two
        NNTest.setInput(number_one, number_two)
        NNTest.forwardPropagation()
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        print(final_error)
        self.assertLess(final_error**2, initial_error**2/10000)

    def test_03_unity_NN_test(self):
        pass

# tester = NeuralNetworkTestSuite()
# tester.test_00_two_number_sum_overlearn_test()
# tester.test_01_two_number_sum_fitness_test()
