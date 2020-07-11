import unittest
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

    def test_02_two_number_sum(self):
        ''' Elementary Neural Network Test. '''
        number_one = 1
        number_two = 2
        desired_number = number_one + number_two
        NNTest = TwoNumberSumNN(number_one, number_two)
        initial_output = NNTest.base_frame.layers['OUTPUT']
        # print(desired_number - initial_output)
        # print(NNTest.base_frame.layers)
        # print(NNTest.base_frame.layers_delta)
        # print(NNTest.weight_base_frame.layers)
        # print([item[:, :, 0] for item in NNTest.weight_base_frame.layers])
        print([item.shape for item in NNTest.weight_base_frame.layers])
        # print(NNTest.weight_base_frame.delta_weights)
        # print(NNTest.weight_base_frame.delta_biases)
        NNTest.runBackpropagation()


tester = NNFrameTestSuite()
# tester.test_01_sigmoid()
tester.test_02_two_number_sum()