from unittest import TestCase
import numpy as np

from mclust.em import *
from mclust.model_factory import ModelFactory, Model

from .utility import apply_resource


class TestMclust(TestCase):
    def setUp(self):
        self.diabetes = apply_resource('data_sets', 'diabetes.csv',
                                       lambda f: np.genfromtxt(f, delimiter=',', skip_header=1))
        self.simulated1d = apply_resource('data_sets', 'simulated1d.csv',
                                          lambda f: np.genfromtxt(f, delimiter=','))
        self.groups = [2, 3, 4, 5]

    def multi_dimensional_test_template(self, model, groups=None):
        if groups is None:
            groups = self.groups
        for group in groups:
            expected = apply_resource('test_data', f'diabetes-{model.value}-{group}-density.csv',
                                      lambda f: np.genfromtxt(f, delimiter=','))

            mod = ModelFactory.create(self.diabetes, model, groups=group)
            mod.fit()
            dens = mod.density(logarithm=True)

            self.assertTrue(np.allclose(expected, dens), f'{model} with {group} groups does not correspond')

    def single_dimensional_test_template(self, model, groups=None):
        if groups is None:
            groups = self.groups
        for group in groups:
            expected = apply_resource('test_data', f'simulated1d-{model.value}-{group}-density.csv',
                                      lambda f: np.genfromtxt(f, delimiter=','))

            if group != 1:
                z = apply_resource('test_data', f'z-diabetes-{group}.csv',
                                   lambda f: np.asfortranarray(np.genfromtxt(f, delimiter=',')))
                mod = ModelFactory.create(self.simulated1d, model, z=z)
            else:
                mod = ModelFactory.create(self.simulated1d, model, groups=1)

            mod.fit()

            dens = mod.density(logarithm=True)

            self.assertTrue(np.allclose(expected, dens), f'{model} with {group} groups does not correspond')

    def test_MEE(self):
        self.single_dimensional_test_template(Model.E)

    def test_MEV(self):
        self.single_dimensional_test_template(Model.V)

    def test_MEVVV(self):
        self.multi_dimensional_test_template(Model.VVV)

    def test_MEEII(self):
        self.multi_dimensional_test_template(Model.EII)

    def test_MEVII(self):
        self.multi_dimensional_test_template(Model.VII)

    def test_MEEEI(self):
        self.multi_dimensional_test_template(Model.EEI)

    def test_MEVEI(self):
        self.multi_dimensional_test_template(Model.VEI)

    def test_MEEVI(self):
        self.multi_dimensional_test_template(Model.EVI)

    def test_MEVVI(self):
        self.multi_dimensional_test_template(Model.VVI)

    def test_MEEVE(self):
        self.multi_dimensional_test_template(Model.EVE)

    def test_MEVEE(self):
        self.multi_dimensional_test_template(Model.VEE)

    def test_MEVVE(self):
        self.multi_dimensional_test_template(Model.VVE)

    def test_MEEEV(self):
        self.multi_dimensional_test_template(Model.EEV)

    def test_MEVEV(self):
        self.multi_dimensional_test_template(Model.VEV)

    def test_MEEVV(self):
        self.multi_dimensional_test_template(Model.EVV)

    def test_MEEEE(self):
        self.multi_dimensional_test_template(Model.EEE)

    def test_MVNX(self):
        self.single_dimensional_test_template(Model.E, [1])

    def test_MVNXII(self):
        self.multi_dimensional_test_template(Model.EII, [1])

    def test_MVNXXI(self):
        self.multi_dimensional_test_template(Model.EEI, [1])

    def test_MVNXXX(self):
        self.multi_dimensional_test_template(Model.EEE, [1])
