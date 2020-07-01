import unittest
import numpy as np
from lib.core.activation.activation import ActivationFunction
from lib.core.activation.sigmoid import SigmoidActivation
from lib.core.activation.tanh import TanhActivation
from lib.core.activation.relu import ReluActivation
from lib.core.activation.leaky_relu import LeakyReluActivation

class ActivationFunctionTestSuite(unittest.TestCase):

    ''' Unit Tests for Activation Functions '''

    def test_00_base(self):
        try:
            ActivationFunction(None, [-0,1,2,3,4,5,6])
        except NotImplementedError as err:
            self.assertTrue('NonImplementation Error Found')
            return
        self.assertFalse('NonImplementation Error Required\n', err)

    def test_01_sigmoid(self):
        test_methods = SigmoidActivation(None, [-0.02,0.01,0.002])
        self.assertTrue('Success')
        if 'float64' in str(type(test_methods.getDerivative())):
            self.assertTrue(True, 'Float Value Received.')
        else:
            self.assertFalse(True, 'Float Value Not Received.')

    def test_02_tanh(self):
        test_methods = TanhActivation(None, [-0.02,0.01,0.002])
        self.assertTrue('Success')
        if 'float64' in str(type(test_methods.getDerivative())):
            self.assertTrue(True, 'Float Value Received.')
        else:
            self.assertFalse(True, 'Float Value Not Received.')

    def test_03_relu(self):
        test_methods = ReluActivation(None, [-0.02,0.01,0.002])
        self.assertTrue('Success')
        if 'float64' in str(type(test_methods.getDerivative())):
            self.assertTrue(True, 'Float Value Received.')
        else:
            self.assertFalse(True, 'Float Value Not Received.')

    def test_04_leaky_relu(self):
        test_methods = LeakyReluActivation(None, [-0.02,0.01,0.002])
        self.assertTrue('Success')
        if 'float64' in str(type(test_methods.getDerivative())):
            self.assertTrue(True, 'Float Value Received.')
        else:
            self.assertFalse(True, 'Float Value Not Received.')

test = ActivationFunctionTestSuite()
test.test_00_base()
test.test_01_sigmoid()
test.test_02_tanh()
test.test_03_relu()
test.test_04_leaky_relu()