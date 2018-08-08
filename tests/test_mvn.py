from unittest import TestCase
import numpy as np

from mclust.multi_var_normal import *


class TestMVN(TestCase):
    test_data_1d = np.array([1,2,3,4,15,16,17,18])
    test_data_2d = (np.arange(24).reshape(3, 8) + 1).transpose()
    test_data = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)])

    def test_fit_X(self):
        model = MVNX(self.test_data_1d)
        model.fit()
        print(model)

    def test_fit_X_dimension(self):
        with self.assertRaises(DimensionError):
            model = MVNX(self.test_data_2d)

    def test_fit_XII(self):
        model = MVNXII(self.test_data_2d)
        model.fit()
        print(model)

    def test_fit_XII_dimension(self):
        with self.assertRaises(DimensionError):
            model = MVNXII(self.test_data_1d)

    def test_fit_XXI(self):
        model = MVNXXI(self.test_data_2d)
        model.fit()
        print(model)
        print(model.variance)

    def test_fit_XXI_dimension(self):
        with self.assertRaises(DimensionError):
            model = MVNXXI(self.test_data_1d)

    def test_fit_XXX(self):
        print(self.test_data)
        model = MVNXXX(self.test_data)
        model.fit()
        print(model)
        # print(model.variance)

    def test_fit_XXX_dimension(self):
        with self.assertRaises(DimensionError):
            model = MVNXXX(self.test_data_1d)
