import unittest

from lib.core.node.frame_reader import FrameReader
from test.core.neural_network.num_by_depth_NN import NumByDepthNN

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
