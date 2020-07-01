import unittest

from lib.core.activation.activation import ActivationFunction
from lib.core.activation.sigmoid import SigmoidActivation
from lib.core.activation.tanh import TanhActivation
from lib.core.activation.relu import ReluActivation
from lib.core.activation.leaky_relu import LeakyReluActivation

class ActivationFunctionTestSuite(unittest.TestCase):

    ''' Unit Tests for Activation Functions '''
    _test_value_ = [-0.02, 0.01, 0.002]

    def test_00_base(self):
        try:
            test_methods = ActivationFunction(None, [-0, 1, 2, 3, 4, 5, 6])
            self.fail('NonImplementation Error Required\n')
        except NotImplementedError:
            try:
                test_methods.getDerivative()
                self.fail('NonImplementation Error Required\n')
            except NotImplementedError:
                return

    def test_01_sigmoid(self):
        test_methods = SigmoidActivation(None, self._test_value_)
        if 'float64' not in str(type(test_methods.getDerivative())):
            self.fail('Float Value Not Received.')

    def test_02_tanh(self):
        test_methods = TanhActivation(None, self._test_value_)
        if 'float64' not in str(type(test_methods.getDerivative())):
            self.fail('Float Value Not Received.')

    def test_03_relu(self):
        test_methods = ReluActivation(None, self._test_value_)
        if 'float64' not in str(type(test_methods.getDerivative())):
            self.fail('Float Value Not Received.')

    def test_04_leaky_relu(self):
        test_methods = LeakyReluActivation(None, self._test_value_)
        if 'float64' not in str(type(test_methods.getDerivative())):
            self.fail('Float Value Not Received.')
