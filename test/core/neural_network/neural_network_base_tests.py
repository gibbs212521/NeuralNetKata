import unittest
from math import isnan
from numpy import average, square, multiply
from numpy.random import random

from test.core.neural_network.two_number_sum_NN import TwoNumberSumNN
from test.core.neural_network.unity_NN import UnityOutputNN
from test.core.neural_network.exp_NN import ExponentNN

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

    def test_02_unity_NN_test(self):
        inputs = multiply(random(1000)-0.5, 100000)
        NNTest = UnityOutputNN(inputs, node_type='RELU')
        avg_initial_variance = average(square(inputs - NNTest.base_frame.layers['OUTPUT']))
        NNTest.trainNN(10)
        NNTest.trainNN(10, learning_rate=0.05)
        final_error = inputs - NNTest.base_frame.layers['OUTPUT']
        for indx, err in enumerate(final_error):
            if isnan(err):
                final_error[indx] = 0
                err = 0
            self.assertLess(err**2, avg_initial_variance/10000)
        # Demonstrate overfitting by handling value outside of scope.
        inputs = multiply(random(1000)-0.75, 100000000)
        NNTest.setInput(inputs)
        NNTest.forwardPropagation()
        final_error = inputs - NNTest.base_frame.layers['OUTPUT']
        for indx, err in enumerate(final_error):
            if isnan(err):
                final_error[indx] = 0
                err = 0
            self.assertLess(err**2, avg_initial_variance/10000)

    def test_03_exponential_test(self):
        inputs = multiply(random(20)-0.5, 150) // 1
        base_number = 13
        power_number = 6
        desired_number = base_number**power_number
        NNTest = ExponentNN(base_number, power_number, node_type='RELU')
        initial_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        NNTest.trainFIRNeuralNetwork(inputs, 2, learning_rate=0.5)
        NNTest.trainFIRNeuralNetwork(inputs, 10, learning_rate=0.1)
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        self.assertLess(final_error**2, initial_error**2/10000)
        # Demonstrate overfitting by handling value outside of scope.
        base_number = -1537
        power_number = -13
        desired_number = base_number**power_number
        NNTest.setInput(base_number, power_number)
        NNTest.forwardPropagation()
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        self.assertLess(final_error**2, initial_error**2/10000)
