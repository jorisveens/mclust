from unittest import TestCase
from .utility import apply_resource

from mclust.hierarchical_clustering import *
from mclust.utility import mclust_unmap


class TestHC(TestCase):
    test_data_2d = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)], float, order='F')
    test_data_sclaed = np.array([[- 1.080123,  - 1.4639385, - 0.5400617],
                                [1.080123, 0.1126107, - 1.6201852],
                                [0.000000, - 1.0134959, - 0.5400617],
                                [- 1.080123, 0.5630533, 0.5400617],
                                [1.080123, - 0.5630533, - 0.5400617],
                                [0.000000, 1.0134959, 0.5400617],
                                [- 1.080123, - 0.1126107, 1.6201852],
                                [1.080123, 1.4639385, 0.5400617]])

    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.test_data_2d = np.array([[(i * j + 4 * (i - j * (.5 * i - 8)) % 12) for i in range(3)] for j in range(8)], float, order='F')
        self.test_data_sclaed = np.array([[- 1.080123,  - 1.4639385, - 0.5400617],
                                         [1.080123, 0.1126107, - 1.6201852],
                                         [0.000000, - 1.0134959, - 0.5400617],
                                         [- 1.080123, 0.5630533, 0.5400617],
                                         [1.080123, - 0.5630533, - 0.5400617],
                                         [0.000000, 1.0134959, 0.5400617],
                                         [- 1.080123, - 0.1126107, 1.6201852],
                                         [1.080123, 1.4639385, 0.5400617]])

    def test_scale(self):
        scaled = scale(self.test_data_2d, center=True, rescale=True)
        self.assertTrue(np.allclose(self.test_data_sclaed, scaled))

    def test_partconv_consec(self):
        test_data = np.array([1, 4, 5, 4, 4, 7, 1, 2, 0])
        expected = np.array([0, 1, 2, 1, 1, 3, 0, 4, 5]) + 1
        result = partconv(test_data, consec=True)
        self.assertTrue(np.array_equal(result, expected))

    def test_partconv_non_consec(self):
        test_data = np.array([1, 4, 5, 4, 4, 7, 1, 2, 0])
        expected = np.array([0, 1, 2, 1, 1, 5, 0, 7, 8]) + 1
        result = partconv(test_data, consec=False)
        self.assertTrue(np.array_equal(result, expected))

    def test_traceW(self):
        result = traceW(self.test_data_2d)
        self.assertEqual(traceW(self.test_data_2d), 450)

    def test_hclass(self):
        model = HCVVV(self.test_data_2d)
        model.fit()
        print(model.get_class_matrix(range(1, 9)))

    def test_hcvvv(self):
        model = HCVVV(self.test_data_2d)
        model.fit()
        print(model.pairs)

    def test_hceii(self):
        model = HCEII(self.test_data_2d)
        model.fit()
        print(model.pairs)

    def test_diabetes(self):
        model = HCVVV(self.diabetes.astype(float, order='F'))
        model.fit()
        print(model.pairs)
        print(model.partition)
        hc_matrix = model.get_class_matrix(range(1, 9))
        print(mclust_unmap(hc_matrix[:, 2]))
