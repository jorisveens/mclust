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

        print(model.predict())
        print(model.classify())
        print(model.density())
        self.assertTrue(np.array_equal(model.predict(), model.classify()))

    def test_predict1d(self):
        model = Mclust(self.simulated1d)
        model.fit()

        print(model.predict())
        print(model.classify())

    def test_iris(self):
        model = Mclust(self.iris[:, range(4)])
        model.fit()
        print(model)
        print(model.predict())
        print(sum(model.predict()))

