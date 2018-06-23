from unittest import TestCase
import numpy as np

from mclust.MVN import MVNXII, MVNXXI, MVNXXX


class TestMVNXXI(TestCase):
    test_data_1d = np.array([1,2,3,4,15,16,17,18])
    test_data_2d = (np.arange(24).reshape(3, 8) + 1).transpose()
    test_data = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)])

    def test_fit_XII(self):
        model = MVNXII()
        model.fit(self.test_data_2d)
        print(model)

    def test_fit_XXI(self):
        model = MVNXXI()
        model.fit(self.test_data_2d)
        print(model)
        print(model.variance)

    def test_fit_XXX(self):
        print(self.test_data)
        model = MVNXXX()
        model.fit(self.test_data)
        print(model)
        # print(model.variance)
