import unittest
import pkg_resources

from mclust.MclustBIC import MclustBIC
from mclust.Models import Model
import numpy as np

# TODO add test for models that fail

resource_package = 'mclust'


def apply_resource(directory, file, func):
    resource_path = '/'.join(('resources', directory, file))
    with pkg_resources.resource_stream(resource_package, resource_path) as f:
        return func(f)


class TestMVN(unittest.TestCase):
    test_data = np.array([1,2,3,4,15,16,17,18])
    test_data_2d = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)])

    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))

    def test_single_dim(self):
        bic = MclustBIC(self.simulated1d)

        # all return codes should be 0
        self.assertTrue(np.all(bic.get_return_codes_matrix() == 0))

        # check if BIC values for default models and groups are the same as in the R version
        expected = apply_resource('test_data', 'simulated1d-BIC.csv',
                                  lambda f: np.asfortranarray(np.genfromtxt(f, delimiter=',', dtype=float)))
        actual = bic.get_bic_matrix()
        self.assertTrue(np.allclose(actual, expected))

        # check if the classification of the best model is correct
        model = bic.pick_best_model()
        expected_classification = apply_resource('test_data', 'simulated1d-best-model-classification.csv',
                                                 lambda f: np.genfromtxt(f, delimiter=','))
        self.assertTrue(np.array_equal(model.classify(), expected_classification))

    def test_multi_dim(self):
        bic = MclustBIC(self.diabetes)

        # all return codes should be 0
        self.assertTrue(np.all(bic.get_return_codes_matrix() == 0))

        # check if BIC values for default models and groups are the same as in the R version
        expected = apply_resource('test_data', 'diabetes-BIC.csv',
                                  lambda f: np.asfortranarray(np.genfromtxt(f, delimiter=',', skip_header=1, dtype=float)))[:, 1:15]
        actual = bic.get_bic_matrix()
        self.assertTrue(np.allclose(actual, expected))

        # check if the classification of the best model is correct
        model = bic.pick_best_model()
        expected_classification = apply_resource('test_data', 'diabetes-best-model-classification.csv',
                                                 lambda f: np.genfromtxt(f, delimiter=','))
        self.assertTrue(np.array_equal(model.classify(), expected_classification))


if __name__ == '__main__':
    unittest.main()
