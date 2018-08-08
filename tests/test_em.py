import unittest
import json
from .utility import apply_resource

from mclust.utility import qclass, mclust_unmap
from mclust.em import *

resource_package = 'mclust'


def clean_json(data):
    unitary = ['modelName', 'n', 'd', 'G', 'loglik', 'returnCode']
    for var in unitary:
        if var not in data.keys():
            continue
        data[var] = data[var][0]

    data['z'] = np.asfortranarray(data['z'])
    parameter_arrays = ['pro', 'mean']
    for var in parameter_arrays:
        data['parameters'][var] = np.asfortranarray(data['parameters'][var])

    if 'sigma' in data['parameters']['variance']:
        data['parameters']['variance']['sigma'] = np.asfortranarray(data['parameters']['variance']['sigma'])
    elif 'sigmasq' in data['parameters']['variance']:
        data['parameters']['variance']['sigmasq'] = np.asfortranarray(data['parameters']['variance']['sigmasq'])
    return data


class METestCase(unittest.TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.groups = [2, 3, 4, 5]
        self.z_matrices = {}
        for group in self.groups:
            self.z_matrices[group] = apply_resource(
                'test_data', f'z-diabetes-{group}.csv',
                lambda f: np.asfortranarray(np.genfromtxt(f,
                                                          delimiter=',')))

    def multi_dimensional_test_template(self, name, func):
        for group in self.groups:
            model = func(self.diabetes, self.z_matrices[group])
            model.fit()
            expected = apply_resource('test_data', f'diabetes-{name}-{group}.json',
                                      lambda f: clean_json(json.load(f)))

            self.assertEqual(expected['returnCode'],
                             model.returnCode)
            self.assertTrue(np.allclose(expected['loglik'], model.loglik))
            self.assertTrue(np.allclose(expected['parameters']['mean'].transpose(),
                                        model.mean))
            self.assertTrue(np.allclose(expected['parameters']['variance']['sigma'].transpose(2, 1, 0),
                                        model.variance.get_covariance()))
            self.assertTrue(np.allclose(expected['z'],
                                        model.z, atol=0.0001))

    def single_dimensional_test_template(self, name, func):
        for group in self.groups:
            model = func(self.simulated1d, self.z_matrices[group])
            model.fit()
            expected = apply_resource('test_data', f'simulated1d-{name}-{group}.json',
                                      lambda f: clean_json(json.load(f)))

            self.assertEqual(expected['returnCode'],
                             model.returnCode)
            self.assertTrue(np.allclose(expected['loglik'], model.loglik))
            self.assertTrue(np.allclose(np.array([expected['parameters']['mean']]).transpose(),
                                        model.mean))
            self.assertTrue(np.allclose(expected['parameters']['variance']['sigmasq'],
                                        model.variance.sigmasq, atol=0.0001))
            self.assertTrue(np.allclose(expected['z'],
                                        model.z, atol=0.0001))

    def test_MEE(self):
        self.single_dimensional_test_template('E', MEE)

    def test_MEV(self):
        self.single_dimensional_test_template('V', MEV)

    def test_MEVVV(self):
        self.multi_dimensional_test_template('VVV', MEVVV)

    def test_MEEII(self):
        self.multi_dimensional_test_template('EII', MEEII)

    def test_MEVII(self):
        self.multi_dimensional_test_template('VII', MEVII)

    def test_MEEEI(self):
        self.multi_dimensional_test_template('EEI', MEEEI)

    def test_MEVEI(self):
        self.multi_dimensional_test_template('VEI', MEVEI)

    def test_MEEVI(self):
        self.multi_dimensional_test_template('EVI', MEEVI)

    def test_MEVVI(self):
        self.multi_dimensional_test_template('VVI', MEVVI)

    def test_MEEVE(self):
        self.multi_dimensional_test_template('EVE', MEEVE)

    def test_MEVEE(self):
        self.multi_dimensional_test_template('VEE', MEVEE)

    def test_MEVVE(self):
        self.multi_dimensional_test_template('VVE', MEVVE)

    def test_MEEEV(self):
        self.multi_dimensional_test_template('EEV', MEEEV)

    def test_MEVEV(self):
        self.multi_dimensional_test_template('VEV', MEVEV)

    def test_MEEVV(self):
        self.multi_dimensional_test_template('EVV', MEEVV)

    def test_MEEEE(self):
        self.multi_dimensional_test_template('EEE', MEEEE)


def random_z(n, g):
    z = np.zeros((n, g), float, order='F')
    for i in range(n):
        sum = 1.0
        for j in range(g - 1):
            rand = np.random.uniform(high=sum)
            z[i, j] = rand
            sum -= rand
        z[i, g - 1] = sum
    return z


class MStepTest(unittest.TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.testData = np.array([1, 20, 3, 34, 5, 0, 7, 12])
        self.groups = [2, 3, 4, 5]
        self.z_matrices = {}
        for group in self.groups:
            self.z_matrices[group] = apply_resource(
                'test_data', f'z-diabetes-{group}.csv',
                lambda f: np.asfortranarray(np.genfromtxt(f,
                                                          delimiter=',')))

    def multi_dimensional_test_template(self, name, func):
        for group in self.groups:
            model = func(self.diabetes, self.z_matrices[group])
            model.fit()
            model.m_step()
            expected = apply_resource('test_data', f'diabetes-{name}-{group}-mstep.json',
                                      lambda f: clean_json(json.load(f)))

            self.assertEqual(expected['returnCode'],
                             model.returnCode)
            self.assertTrue(np.allclose(expected['parameters']['mean'].transpose(),
                                        model.mean))
            self.assertTrue(np.allclose(expected['parameters']['variance']['sigma'].transpose(2, 1, 0),
                                        model.variance.get_covariance()))

    def test_mstepE(self):
        z = mclust_unmap(qclass(self.testData, 3))
        model = MEE(self.testData, z)
        model.fit()
        print(model)
        model.m_step()
        print(model)

    def test_mstepV(self):
        z = mclust_unmap(qclass(self.testData, 3))
        model = MEV(self.testData, z)
        model.fit()
        print(model)
        model.m_step()
        print(model)

    def test_mstepEEE(self):
        self.multi_dimensional_test_template('EEE', MEEEE)

    def test_mstepEII(self):
        self.multi_dimensional_test_template('EII', MEEII)

    def test_mstepVII(self):
        self.multi_dimensional_test_template('VII', MEVII)

    def test_mstepEEI(self):
        self.multi_dimensional_test_template('EEI', MEEEI)

    def test_mstepVEI(self):
        self.multi_dimensional_test_template('VEI', MEVEI)

    def test_mstepEVI(self):
        self.multi_dimensional_test_template('EVI', MEEVI)

    def test_mstepVVI(self):
        self.multi_dimensional_test_template('VVI', MEVVI)

    def test_mstepEVE(self):
        self.multi_dimensional_test_template('EVE', MEEVE)

    def test_mstepVEE(self):
        self.multi_dimensional_test_template('VEE', MEVEE)

    def test_mstepVVE(self):
        self.multi_dimensional_test_template('VVE', MEVVE)

    def test_mstepEEV(self):
        self.multi_dimensional_test_template('EEV', MEEEV)

    def test_mstepVEV(self):
        self.multi_dimensional_test_template('VEV', MEVEV)

    def test_mstepEVV(self):
        self.multi_dimensional_test_template('EVV', MEEVV)

    def test_mstepVVV(self):
        self.multi_dimensional_test_template('VVV', MEVVV)


class EStepTest(unittest.TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', "diabetes.csv",
                                       lambda f: np.genfromtxt(f, skip_header=1, delimiter=','))
        self.classes = apply_resource('data_sets', "diabetes_classification.csv",
                                      lambda f: np.genfromtxt(f, delimiter=','))

    def multi_dimensional_test_template(self, name, func):
        model = func(self.diabetes, mclust_unmap(self.classes))
        model.m_step()
        model.e_step()
        expected = apply_resource('test_data', f'diabetes-{name}-3-estep.json',
                                  lambda f: clean_json(json.load(f)))

        self.assertEqual(expected['returnCode'],
                         model.returnCode)
        self.assertTrue(np.allclose(expected['parameters']['mean'].transpose(),
                                    model.mean))
        self.assertTrue(np.allclose(expected['parameters']['variance']['sigma'].transpose(2, 1, 0),
                                    model.variance.get_covariance()))
        self.assertTrue(np.allclose(expected['loglik'], model.loglik))
        self.assertTrue(np.allclose(expected['z'], model.z, atol=0.0001))

    def test_mstepEEE(self):
        self.multi_dimensional_test_template('EEE', MEEEE)

    def test_mstepEII(self):
        self.multi_dimensional_test_template('EII', MEEII)

    def test_mstepVII(self):
        self.multi_dimensional_test_template('VII', MEVII)

    def test_mstepEEI(self):
        self.multi_dimensional_test_template('EEI', MEEEI)

    def test_mstepVEI(self):
        self.multi_dimensional_test_template('VEI', MEVEI)

    def test_mstepEVI(self):
        self.multi_dimensional_test_template('EVI', MEEVI)

    def test_mstepVVI(self):
        self.multi_dimensional_test_template('VVI', MEVVI)

    def test_mstepEVE(self):
        self.multi_dimensional_test_template('EVE', MEEVE)

    def test_mstepVEE(self):
        self.multi_dimensional_test_template('VEE', MEVEE)

    def test_mstepVVE(self):
        self.multi_dimensional_test_template('VVE', MEVVE)

    def test_mstepEEV(self):
        self.multi_dimensional_test_template('EEV', MEEEV)

    def test_mstepVEV(self):
        self.multi_dimensional_test_template('VEV', MEVEV)

    def test_mstepEVV(self):
        self.multi_dimensional_test_template('EVV', MEEVV)

    def test_mstepVVV(self):
        self.multi_dimensional_test_template('VVV', MEVVV)


if __name__ == '__main__':
    unittest.main()
