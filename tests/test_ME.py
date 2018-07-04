import unittest
import numpy as np

from mclust.Utility import qclass, unmap
from mclust.ME import *
from mclust.MclustBIC import random_z


class MyTestCase(unittest.TestCase):
    testData = np.array([1, 20, 3, 34, 5, 0, 7, 12])
    test_data = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)], float, order='F')
    # TODO pacakge test data
    diabetes = np.genfromtxt("/home/joris/Documents/UCD/final_project/diabetes.csv", delimiter=',', skip_header=1)

    def test_MEE(self):
        z = unmap(qclass(self.testData, 2))
        print(self.testData.shape)
        me = MEE(self.testData)
        me.fit(z)
        print(me.variance.get_covariance())
        print(me)

    def test_MEV(self):
        z = unmap(qclass(self.testData, 2))
        me = MEV(self.testData)
        me.fit(z)
        print(me.variance.get_covariance())
        print(me)

    def test_MEVVV(self):
        z = random_z(self.diabetes.shape[0], 3)
        model = MEVVV(self.diabetes)
        print(model.fit(z))
        print(model)

    def test_MEEEE(self):
        z = random_z(self.diabetes.shape[0], 7)
        model = MEEEE(self.diabetes)
        model.fit(z)
        print(model)


if __name__ == '__main__':
    unittest.main()
