from unittest import TestCase
import numpy as np

from mclust.mclust import Mclust, Model

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))

    def test_diabetes(self):
        model = Mclust(self.diabetes)
        model.fit()

        self.assertEqual(model.model, Model.VVV)
        self.assertEqual(model.G, 3)
        self.assertAlmostEqual(model.loglik, -2303.4955606441354)
