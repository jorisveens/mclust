from unittest import TestCase
import numpy as np

from mclust.mclust_da import MclustDA
from mclust.exceptions import ModelError

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        pass

    def test_handle_input_correct(self):
        data = np.zeros((145, 3))
        model = MclustDA(data, np.array([1, 2, 4]))
        self.assertEqual(len(model.g), 3)
        self.assertTrue(np.array_equal(model.g[0], np.arange(1, 6)))

        model = MclustDA(data, np.array([1, 3, 4]), g=[1, 3, 4, 5])
        self.assertTrue(len(model.g), 3)
        self.assertTrue(np.array_equal(model.g[0], [1, 3, 4, 5]))

        g = {0: np.array([2, 3, 4, 1], dtype=int),
             1: np.array([1, 2, 3], dtype=int)}
        model = MclustDA(data, np.array([1, 3]), g)
        self.assertEqual(len(model.g), 2)
        self.assertTrue(np.array_equal(model.g[0], np.array([1, 2, 3, 4])))
        self.assertTrue(np.array_equal(model.g[1], np.array([1, 2, 3])))

    def test_handle_input_failure(self):
        data = np.zeros((145, 3))
        with self.assertRaises(ModelError):
            MclustDA(data, np.array([1, 2, 3]), g=np.array([0, 1, 2, 3]))


