import unittest

from numpy import array, round as round

from lib.core.activation.activation import ActivationFunction
from lib.core.activation.sigmoid import SigmoidActivation
from lib.core.activation.tanh import TanhActivation
from lib.core.activation.relu import ReluActivation
from lib.core.activation.leaky_relu import LeakyReluActivation

class ActivationFunctionTestSuite(unittest.TestCase):

    ''' Unit Tests for Activation Functions '''
    _test_value_ = [-0.02, 0.01, 0.002]
    _test_vector_value = array([0, -1000, 1000])

    def test_00_base(self):
        try:
            ActivationFunction([-0, 1, 2, 3, 4, 5, 6])
            self.fail('NonImplementation Error Required\n')
        except NotImplementedError:
            return

    def test_01_sigmoid(self):
        test_method = SigmoidActivation
        method_test = test_method(self._test_value_)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')
        input_value = 0
        expected_value = 0.5
        test_value = test_method(input_value).resultant
        self.assertEqual(test_value, expected_value)
        input_value = 1000
        expected_value = 1
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)
        input_value = -1000
        expected_value = 0
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)

    def test_02_tanh(self):
        test_method = TanhActivation
        method_test = test_method(self._test_value_)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')
        input_value = 0
        expected_value = 0
        test_value = test_method(input_value).resultant
        self.assertEqual(test_value, expected_value)
        input_value = 1000
        expected_value = 1
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)
        input_value = -1000
        expected_value = -1
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)

    def test_03_relu(self):
        test_method = ReluActivation
        method_test = test_method(self._test_value_)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')
        input_value = 0
        expected_value = 0
        test_value = test_method(input_value).resultant
        self.assertEqual(test_value, expected_value)
        input_value = 1000
        expected_value = 1000
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)
        input_value = -1000
        expected_value = 0
        test_value = round(test_method(input_value).resultant, 10)
        self.assertEqual(test_value, expected_value)

    def test_04_leaky_relu(self):
        test_method = LeakyReluActivation
        method_test = test_method(self._test_value_)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')
        input_value = 0
        expected_value = 0
        test_value = test_method(input_value).resultant
        self.assertEqual(test_value, expected_value)
        input_value = 1000
        expected_value = 1000
        method_test = test_method(input_value)
        test_value = round(method_test.resultant, 10)
        self.assertEqual(test_value, expected_value)
        input_value = -1000
        method_test = test_method(input_value)
        expected_value = -1000 * method_test.negative_ratio
        test_value = round(method_test.resultant, 10)
        self.assertEqual(test_value, expected_value)

    def test_05_sigmoid_vector(self):
        test_method = SigmoidActivation
        method_test = test_method(self._test_vector_value)
        expected_value = [0.5, 1, 0]
        test_value = round(method_test.resultant, 10)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')

        # self.assertEqual(test_value, expected_value)

    def test_06_tanh_vector(self):
        test_method = TanhActivation
        method_test = test_method(self._test_vector_value)
        expected_value = [0, 1, -1]
        test_value = round(method_test.resultant, 10)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')

    def test_07_relu_vector(self):
        test_method = ReluActivation
        method_test = test_method(self._test_vector_value)
        expected_value = [0, 1000, 0]
        test_value = round(method_test.resultant, 10)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')

    def test_08_leaky_relu_vector(self):
        test_method = LeakyReluActivation
        method_test = test_method(self._test_vector_value)
        expected_value = [0, 1000, -1000 * method_test.negative_ratio]
        test_value = round(method_test.resultant, 10)
        if 'float64' not in str(type(method_test.getDerivative())):
            self.fail('Float Value Not Received.')
