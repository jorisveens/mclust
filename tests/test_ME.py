import unittest
import numpy as np

from mclust.Utility import qclass, mclust_unmap
from mclust.hc import HCVVV
from mclust.ME import *


class METestCase(unittest.TestCase):
    testData = np.array([1, 20, 3, 34, 5, 0, 7, 12])
    test_data = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)], float, order='F')
    # TODO pacakge test data
    diabetes = np.genfromtxt("/home/joris/Documents/UCD/final_project/diabetes.csv", delimiter=',', skip_header=1)

    def test_MEE(self):
        z = mclust_unmap(qclass(self.testData, 2))
        print(self.testData.shape)
        me = MEE(self.testData)
        me.fit(z)
        print(me.variance.get_covariance())
        print(me)

    def test_MEV(self):
        z = mclust_unmap(qclass(self.testData, 2))
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


def random_z(n, g):
    z = np.zeros((n, g), float, order='F')
    for i in range(n):
        sum = 1.0
        for j in range(g-1):
            rand = np.random.uniform(high=sum)
            z[i, j] = rand
            sum -= rand
        z[i, g-1] = sum
    return z


class MStepTest(unittest.TestCase):
    testData = np.array([1, 20, 3, 34, 5, 0, 7, 12])
    diabetes = np.genfromtxt("/home/joris/Documents/UCD/final_project/diabetes.csv", delimiter=',', skip_header=1)

    def test_mstepE(self):
        z = mclust_unmap(qclass(self.testData, 3))
        model = MEE(self.testData)
        model.fit(z)
        print(model.z)
        print(model)
        model.m_step()
        print(model.z)
        print(model)

    def test_mstepV(self):
        z = mclust_unmap(qclass(self.testData, 3))
        model = MEV(self.testData)
        model.fit(z)
        print(model.z)
        print(model)
        model.m_step()
        print(model.z)
        print(model)

    def test_mstepEEE(self):
        hc = HCVVV(self.diabetes)
        hc.fit()
        z = mclust_unmap(hc.get_class_matrix([3])[:, 0])
        model = MEEEE(self.diabetes)
        model.fit(z)
        print(model.z)
        print(model)
        model.m_step()
        print(model.z)
        print(model)

    def test_mstepVVV(self):
        hc = HCVVV(self.diabetes)
        hc.fit()
        z = mclust_unmap(hc.get_class_matrix([3])[:, 0])
        model = MEVVV(self.diabetes)
        model.fit(z)
        print(model.z)
        print(model)
        model.m_step()
        print(model.z)
        print(model)


if __name__ == '__main__':
    unittest.main()
