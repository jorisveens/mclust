import unittest

from mclust.BIC import mclustBIC
from mclust.Models import Model
import numpy as np


class TestMVN(unittest.TestCase):
    test_data = np.array([1,2,3,4,15,16,17,18])
    test_data_2d = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)])
    diabetes = np.genfromtxt("/home/joris/Documents/UCD/final_project/diabetes.csv", delimiter=',', skip_header=1)

    def test_BIC_E(self):
        out = mclustBIC(self.test_data, models=[Model.E])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_BIC_V(self):
        out = mclustBIC(self.test_data, models=[Model.V])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_BIC_1D(self):
        out = mclustBIC(self.test_data)
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)
        bestgroup, bestmodel = map(lambda a: a.tolist()[0], np.where(out.BICMatrix == np.nanmax(out.BICMatrix)))
        print(f"best model: {out.models[bestmodel].name}, {out.groups[bestgroup]},\n \
              {out.BICMatrix[bestgroup, bestmodel]}")

    def test_BIC_X(self):
        out = mclustBIC(self.test_data, models=[Model.X])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_BIC_EII(self):
        print(self.test_data_2d)
        out = mclustBIC(self.test_data_2d, g=[1], models=[Model.EII])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_BIC_VVV(self):
        out = mclustBIC(self.diabetes, models=[Model.VVV])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_BIC_EEE(self):
        out = mclustBIC(self.diabetes, models=[Model.EEE])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)

    def test_multi_dim(self):
        out = mclustBIC(self.diabetes, models=[Model.EEE, Model.VVV])
        print(out.BICMatrix)
        print(out.returnCodes)
        print(out.models)
        print(out.groups)


if __name__ == '__main__':
    unittest.main()
