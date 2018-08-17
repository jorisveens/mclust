import unittest
import pkg_resources

from mclust.bic import MclustBIC
from mclust.models import Model
import numpy as np
resource_package = 'mclust'


def apply_resource(directory, file, func):
    resource_path = '/'.join(('resources', directory, file))
    with pkg_resources.resource_stream(resource_package, resource_path) as f:
        return func(f)


class TestBIC(unittest.TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.iris = apply_resource('data_sets', 'iris.csv',
                                   lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))

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
        self.assertTrue(np.array_equal(model.predict(), expected_classification))

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
        self.assertTrue(np.array_equal(model.predict(), expected_classification))

    def test_iris(self):
        bic = MclustBIC(self.iris[:, range(4)])

        expected_return = apply_resource('test_data', 'iris-BIC-return.csv',
                                         lambda f: np.genfromtxt(f, delimiter=','))
        expected_bic = apply_resource('test_data', 'iris-BIC.csv',
                                      lambda f: np.genfromtxt(f, delimiter=','))

        self.assertTrue(np.array_equal(bic.get_return_codes_matrix(), expected_return))
        self.assertTrue(np.allclose(bic.get_bic_matrix(), expected_bic, equal_nan=True))

        model = bic.pick_best_model()
        self.assertEqual(model.model, Model.VEV)
        self.assertEqual(model.g, 2)


if __name__ == '__main__':
    unittest.main()
