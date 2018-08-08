from unittest import TestCase
import numpy as np

from mclust.classification import MclustDA, EDDA, DiscriminantAnalysis
from mclust.models import Model
from mclust.exceptions import ModelError

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', "diabetes.csv",
                                       lambda f: np.genfromtxt(f, skip_header=1, delimiter=','))
        self.diabets_cls = apply_resource('data_sets', "diabetes_classification.csv",
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.iris = apply_resource('data_sets', 'iris.csv',
                                   lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))

    def test_handle_input_correct(self):
        data = np.array([12, 13, 56, 60])
        da = DiscriminantAnalysis(data, np.array([1, 1, 2, 2]))
        self.assertEqual(len(da.g), 2)
        self.assertTrue(np.array_equal(da.g[0], np.arange(1, 6)))

        da = DiscriminantAnalysis(data, np.array([1, 3, 4]), g=[1, 3, 4, 5])
        self.assertTrue(len(da.g), 3)
        self.assertTrue(np.array_equal(da.g[0], [1, 3, 4, 5]))

        g = {0: np.array([2, 3, 4, 1], dtype=int),
             1: np.array([1, 2, 3], dtype=int)}
        da = DiscriminantAnalysis(data, np.array([1, 3]), g)
        self.assertEqual(len(da.g), 2)
        self.assertTrue(np.array_equal(da.g[0], np.array([1, 2, 3, 4])))
        self.assertTrue(np.array_equal(da.g[1], np.array([1, 2, 3])))

    def test_handle_input_failure(self):
        data = np.zeros((145, 3))
        with self.assertRaises(ModelError):
            DiscriminantAnalysis(data, np.array([1, 2, 3]), g=np.array([0, 1, 2, 3]))

    def test_edda(self):
        da = EDDA(self.diabetes, self.diabets_cls)
        self.assertEqual(len(da.fitted_models), 3)
        self.assertEqual(da.fitted_models[0].model, Model.VVV)
        self.assertEqual(da.fitted_models[1].model, Model.VVV)
        self.assertEqual(da.fitted_models[2].model, Model.VVV)

        self.assertEqual(da.fitted_models[0].g, 1)
        self.assertEqual(da.fitted_models[1].g, 1)
        self.assertEqual(da.fitted_models[2].g, 1)

    def test_edda_prediction(self):
        da = EDDA(self.diabetes, self.diabets_cls)
        pred = da.predict()

        expected = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
             1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2,
             2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        self.assertTrue(np.array_equal(pred, expected))

    def test_mclust_da(self):
        da = MclustDA(self.diabetes, self.diabets_cls)
        self.assertEqual(len(da.fitted_models), 3)
        self.assertEqual(da.fitted_models[0].model, Model.EEI)
        self.assertEqual(da.fitted_models[1].model, Model.EVI)
        self.assertEqual(da.fitted_models[2].model, Model.VVV)

        self.assertEqual(da.fitted_models[0].g, 4)
        self.assertEqual(da.fitted_models[1].g, 2)
        self.assertEqual(da.fitted_models[2].g, 3)

    def test_mclust_da_prediction(self):
        da = MclustDA(self.diabetes, self.diabets_cls)
        pred = da.predict()

        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                             1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertTrue(np.array_equal(pred, expected))

    def test_edda_iris(self):
        da = EDDA(self.iris[:, range(4)], self.iris[:, 4])
        self.assertEqual(len(da.fitted_models), 3)
        self.assertEqual(da.fitted_models[0].model, Model.VEV)
        self.assertEqual(da.fitted_models[1].model, Model.VEV)
        self.assertEqual(da.fitted_models[2].model, Model.VEV)

        self.assertEqual(da.fitted_models[0].g, 1)
        self.assertEqual(da.fitted_models[1].g, 1)
        self.assertEqual(da.fitted_models[2].g, 1)

    def test_edda_iris_prediction(self):
        da = EDDA(self.iris[:, range(4)], self.iris[:, 4])
        pred = da.predict()

        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
             2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2,
             2, 2])

        self.assertTrue(np.array_equal(pred, expected))

    def test_mclust_da_iris(self):
        da = MclustDA(self.iris[:, range(4)], self.iris[:, 4])
        self.assertEqual(len(da.fitted_models), 3)
        self.assertEqual(da.fitted_models[0].model, Model.EEE)
        self.assertEqual(da.fitted_models[1].model, Model.XXX)
        self.assertEqual(da.fitted_models[2].model, Model.XXX)

        self.assertEqual(da.fitted_models[0].g, 2)
        self.assertEqual(da.fitted_models[1].g, 1)
        self.assertEqual(da.fitted_models[2].g, 1)

    def test_mclust_da_iris_prediction(self):
        da = MclustDA(self.iris[:, range(4)], self.iris[:, 4])
        pred = da.predict()

        expected = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2,
             2, 2])

        self.assertTrue(np.array_equal(pred, expected))

    def test_edda_loglik(self):
        da = EDDA(self.diabetes, self.diabets_cls)
        loglik = da.loglik()
        self.assertTrue(np.isclose(loglik, -2328.979))

        da = EDDA(self.iris[:, range(4)], self.iris[:, 4])
        loglik = da.loglik()
        self.assertTrue(np.isclose(loglik, -187.7097))

    def test_mclust_da_loglik(self):
        da = MclustDA(self.diabetes, self.diabets_cls)
        loglik = da.loglik()
        self.assertTrue(np.isclose(loglik, -2264.37, atol=0.01))

        da = MclustDA(self.iris[:, range(4)], self.iris[:, 4])
        loglik = da.loglik()
        self.assertTrue(np.isclose(loglik, -172.8135))
