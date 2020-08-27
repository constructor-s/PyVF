"""

Copyright 2020 Bill Runjie Shi
At the Vision and Eye Movements Lab, University of Toronto.
Visit us at: http://www.eizenman.ca/

This file is part of PyVF.

PyVF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyVF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyVF. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class BayesianLinearRegression:
    def __init__(self, measurement_std, slope_std, intercept_std):
        self.measurement_std = measurement_std
        self.slope_std = slope_std
        self.intercept_std = intercept_std

    def fit(self, xdata, ydata):
        """
        References
        ----------------
        https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec19-slides.pdf
        """

        # Center the data around mean of x and mean of y, but without scaling to unit variance,
        # such that the provided bayesian prior does not need to be modified
        self.xscalar = StandardScaler(with_mean=True, with_std=False).fit(xdata.reshape(-1, 1))
        self.yscalar = StandardScaler(with_mean=True, with_std=False).fit(ydata.reshape(-1, 1))

        # Scaled values:
        x = self.xscalar.transform(xdata.reshape(-1, 1)).ravel()
        y = self.yscalar.transform(ydata.reshape(-1, 1)).ravel()

        # Rename some variables locally
        sig = self.measurement_std  # Data measurement standard deviation
        S = np.array([[self.slope_std**2, 0], [0, self.intercept_std**2]])  # Bayesian prior of covariance of weights
        S_inv = np.linalg.pinv(S)

        Phi = x.reshape(-1, 1)
        Phi = np.hstack([Phi, np.ones_like(Phi)])  # Feature vector

        Sigma_inv = sig ** (-2) * Phi.T @ Phi + S_inv
        Sigma = np.linalg.pinv(Sigma_inv)

        Mu = sig ** (-2) * Sigma @ Phi.T @ y

        self.Sigma = Sigma
        self.Mu = Mu

    def _get_sigma2_pred(self, xdata):
        sig = self.measurement_std
        x = self.xscalar.transform(xdata.reshape(-1, 1)).ravel()
        Phi = x.reshape(-1, 1)
        Phi = np.hstack([Phi, np.ones_like(Phi)])  # Feature vector
        sigma2_pred = np.diag(Phi @ self.Sigma @ Phi.T + sig * sig)
        return sigma2_pred

    def predict(self, xdata):
        x = self.xscalar.transform(xdata.reshape(-1, 1)).ravel()
        Phi = x.reshape(-1, 1)
        Phi = np.hstack([Phi, np.ones_like(Phi)])  # Feature vector

        y_mean = self.Mu @ Phi.T
        y_mean = self.yscalar.inverse_transform(y_mean)

        y_std = np.sqrt(self._get_sigma2_pred(xdata))

        return y_mean, y_std



