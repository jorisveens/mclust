from unittest import TestCase
import numpy as np

from mclust.mclust import Mclust, Model

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.iris = apply_resource('data_sets', 'iris.csv',
                                   lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))

    def test_diabetes(self):
        model = Mclust(self.diabetes)
        model.fit()

        self.assertEqual(model.model, Model.VVV)
        self.assertEqual(model.g, 3)
        self.assertAlmostEqual(model.loglik, -2303.4955606441354)

    def test_predict(self):
        model = Mclust(self.diabetes)
        model.fit()
        expected = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1,
             1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 2,
             2, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
             3]) - 1

        self.assertTrue(np.array_equal(model.predict(), expected))

    def test_predict1d(self):
        model = Mclust(self.simulated1d)
        model.fit()
        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 3, 1,
             2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2,
             2, 3, 2, 2, 1, 2, 1, 2, 1, 3, 2, 2, 2, 1, 2, 2, 3, 2, 2, 1, 3, 1, 3, 1, 3, 2, 2, 1, 3, 4, 3, 4, 3, 3, 3, 4,
             3, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 3, 4, 3, 3, 4, 4, 3, 4, 3, 4, 3, 3,
             3])

        self.assertTrue(np.array_equal(model.predict(), expected))

    def test_iris(self):
        model = Mclust(self.iris[:, range(4)])
        model.fit()

        self.assertEqual(model.model, Model.VEV)
        self.assertEqual(model.g, 2)
        self.assertTrue(np.isclose(model.loglik, -215.726))

        expected = np.concatenate((np.zeros(50), np.ones(100)))

        self.assertTrue(np.array_equal(model.predict(), expected))
