from unittest import TestCase
import numpy as np

from mclust.mclust_da import MclustDA, EDDA, DiscriminantAnalysis
from mclust.exceptions import ModelError

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        pass

    def test_handle_input_correct(self):
        data = np.array([12, 13, 56, 60])
        model = DiscriminantAnalysis(data, np.array([1, 1, 2, 2]))
        self.assertEqual(len(model.g), 2)
        self.assertTrue(np.array_equal(model.g[0], np.arange(1, 6)))

        model = DiscriminantAnalysis(data, np.array([1, 3, 4]), g=[1, 3, 4, 5])
        self.assertTrue(len(model.g), 3)
        self.assertTrue(np.array_equal(model.g[0], [1, 3, 4, 5]))

        g = {0: np.array([2, 3, 4, 1], dtype=int),
             1: np.array([1, 2, 3], dtype=int)}
        model = DiscriminantAnalysis(data, np.array([1, 3]), g)
        self.assertEqual(len(model.g), 2)
        self.assertTrue(np.array_equal(model.g[0], np.array([1, 2, 3, 4])))
        self.assertTrue(np.array_equal(model.g[1], np.array([1, 2, 3])))

    def test_handle_input_failure(self):
        data = np.zeros((145, 3))
        with self.assertRaises(ModelError):
            DiscriminantAnalysis(data, np.array([1, 2, 3]), g=np.array([0, 1, 2, 3]))

    def test_edda(self):
        data = apply_resource('data_sets', "diabetes.csv",
                              lambda f: np.genfromtxt(f, skip_header=1, delimiter=','))
        classes = apply_resource('data_sets', "diabetes_classification.csv",
                                 lambda f: np.genfromtxt(f, delimiter=','))

        model = EDDA(data, classes)

        print(model.predict())
        print(list(model.n))

    def test_mclust_da(self):
        data = apply_resource('data_sets', "diabetes.csv",
                              lambda f: np.genfromtxt(f, skip_header=1, delimiter=','))
        classes = apply_resource('data_sets', "diabetes_classification.csv",
                                 lambda f: np.genfromtxt(f, delimiter=','))

        model = MclustDA(data, classes)
        for mod in model.fitted_models.values():
            print(mod)

        print(model.predict())

    def test_loglik(self):
        data = apply_resource('data_sets', "diabetes.csv",
                              lambda f: np.genfromtxt(f, skip_header=1, delimiter=','))
        classes = apply_resource('data_sets', "diabetes_classification.csv",
                                 lambda f: np.genfromtxt(f, delimiter=','))

        model = MclustDA(data, classes)
        print(model.loglik())
        print(model.df())

        model = EDDA(data, classes)
        print(model.loglik())
        print(model.df())






