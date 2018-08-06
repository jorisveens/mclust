import warnings
import numpy as np


# TODO maybe put as function in EMMixtureModel class
class CDens:
    def __init__(self, data, model, logarithm=False):
        ret = self._handle_input(data, model, logarithm)
        if ret != 0:
            return

        model.pro = np.array([-1], float, order='F')
        model.e_step()

        if model.return_code == -1 or model.return_code == -9:
            model.z = None
        elif model.return_code == 0:
            if not logarithm:
                model.z = np.exp(model.z)

    def _handle_input(self, data, model, logarithm):
        self.data = data,
        self.logarithm = logarithm
        self.n, self.d = data.shape
        self.G = len(model.mean)
        self.z = None
        self.returnCode = None

        if model.mean is None or model.pro is None or model.variance is None:
            warnings.warn("parameters are missing")
            self.z = np.fill_diagonal((self.n, self.G), float('nan'))
            self.returnCode = 9
            return 9

        return 0




