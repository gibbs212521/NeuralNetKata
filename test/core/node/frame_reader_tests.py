import unittest
from math import isnan
import numpy as np

from lib.core.node.frame_reader import FrameReader
from test.core.node.num_by_depth_NN import NumByDepthNN
from test.core.node.two_number_sum_NN import TwoNumberSumNN

class NNFrameTestSuite(unittest.TestCase):

    ''' Unit Tests for Activation Functions '''
    _test_value_ = [-0.02, 0.01, 0.002]

    def test_00_base(self):
        try:
            FrameReader()
            self.fail('NonImplementation Error Required of define_frame\n')
        except NotImplementedError:
            NumByDepthNN(1, 1, 'SIGMOID')
            NumByDepthNN(1, 1, node_type='TANH')
            NumByDepthNN(1, 1, node_type='RELU')
            NumByDepthNN(1, 1, node_type='LEAKY_RELU')
            return

    def test_01_sigmoid(self):
        num_hidden_layers = 4
        nodes_per_hidden_layer = 4
        NNTest = NumByDepthNN(num_hidden_layers, nodes_per_hidden_layer, node_type='SIGMOID')
        self.assertEqual(len(NNTest.base_frame.layers), num_hidden_layers+2)

    def test_02_two_number_sum_overlearn_test(self):
        ''' Elementary Neural Network Test. '''
        number_one = 1
        number_two = 2
        desired_number = number_one + number_two
        NNTest = TwoNumberSumNN(number_one, number_two)
        initial_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        for k in range(10):
            err = desired_number - NNTest.base_frame.layers['OUTPUT']
            NNTest.runBackpropagation(err)
            NNTest.forwardPropagation()
        NNTest.learning_rate = 0.05
        for k in range(100):
            err = desired_number - NNTest.base_frame.layers['OUTPUT']
            NNTest.runBackpropagation(err)
            NNTest.forwardPropagation()
        NNTest.learning_rate = 0.001
        for k in range(100):
            err = desired_number - NNTest.base_frame.layers['OUTPUT']
            NNTest.runBackpropagation(err)
            NNTest.forwardPropagation()
        final_error = desired_number - NNTest.base_frame.layers['OUTPUT']
        if isnan(final_error):
            final_error = 0
        self.assertLess(final_error**2, initial_error**2/100)




tester = NNFrameTestSuite()
# tester.test_01_sigmoid()
tester.test_02_two_number_sum()