from unittest import TestCase
from utility import apply_resource

from mclust.multi_var_normal import *


class TestMVN(TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.iris = apply_resource('data_sets', 'iris.csv',
                                   lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))

    def test_fit_X(self):
        model = MVNX(self.simulated1d)
        model.fit()

        self.assertTrue(np.isclose(model.mean[0, 0], 10.60357))
        self.assertTrue(np.isclose(model.loglik, -435.6605))
        self.assertTrue(np.isclose(model.variance.get_covariance()[0, 0], 23.83685))

    def test_fit_X_dimension(self):
        with self.assertRaises(DimensionError):
            MVNX(self.diabetes)

    def test_fit_XII(self):
        model = MVNXII(self.diabetes)
        model.fit()

        self.assertTrue(np.allclose(model.mean, [[121.9862, 540.7862, 186.1172]]))
        self.assertTrue(np.isclose(model.loglik, -2922.008))
        self.assertTrue(np.allclose(model.variance.get_covariance(),
                                    [[[40000.28, 0, 0],
                                      [0, 40000.28, 0],
                                      [0, 0, 40000.28]]]))

    def test_fit_XII_dimension(self):
        with self.assertRaises(DimensionError):
            MVNXII(self.simulated1d)

    def test_fit_XXI(self):
        model = MVNXXI(self.diabetes)
        model.fit()

        self.assertTrue(np.allclose(model.mean, [[121.9862, 540.7862, 186.1172]]))
        self.assertTrue(np.isclose(model.loglik, -2750.135))
        self.assertTrue(np.allclose(model.variance.get_covariance(),
                                    [[[4058.91, 0, 0],
                                     [0, 101417.5, 0],
                                     [0, 0, 14524.45]]]))

    def test_fit_XXI_dimension(self):
        with self.assertRaises(DimensionError):
            MVNXXI(self.simulated1d)

    def test_fit_XXX(self):
        model = MVNXXX(self.diabetes)
        model.fit()

        self.assertTrue(np.allclose(model.mean, [[121.9862, 540.7862, 186.1172]]))
        self.assertTrue(np.isclose(model.loglik, -2545.828))
        self.assertTrue(np.allclose(model.variance.get_covariance(),
                                    [[[4058.910, 19544.12, -3042.336],
                                      [19544.121, 101417.48, -13411.223],
                                      [-3042.336, -13411.22, 14524.448]]]))

    def test_fit_XXX_dimension(self):
        with self.assertRaises(DimensionError):
            MVNXXX(self.simulated1d)
