import unittest

from mclust.MclustBIC import MclustBIC
from mclust.Models import Model
import numpy as np


class TestMVN(unittest.TestCase):
    test_data = np.array([1,2,3,4,15,16,17,18])
    test_data_2d = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)])
    diabetes = np.genfromtxt("/home/joris/Documents/UCD/final_project/diabetes.csv", delimiter=',', skip_header=1)

    def test_BIC_E(self):
        out = MclustBIC(self.test_data, models=[Model.E])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_BIC_V(self):
        out = MclustBIC(self.test_data, models=[Model.V])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_BIC_1D(self):
        out = MclustBIC(self.test_data)
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)
        model = out.pick_best_model()
        print(model)
        print(model.classify())

    def test_BIC_X(self):
        out = MclustBIC(self.test_data, models=[Model.X])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_BIC_EII(self):
        print(self.test_data_2d)
        out = MclustBIC(self.test_data_2d, groups=[1], models=[Model.EII])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_BIC_VVV(self):
        out = MclustBIC(self.diabetes, models=[Model.VVV])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_BIC_EEE(self):
        out = MclustBIC(self.diabetes, models=[Model.EEE])
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)

    def test_multi_dim(self):
        out = MclustBIC(self.diabetes)
        print(out.get_bic_matrix())
        print(out.get_return_codes_matrix())
        print(out.models)
        print(out.groups)
        model = out.pick_best_model()
        print(model)
        print(model.classify())


if __name__ == '__main__':
    unittest.main()
