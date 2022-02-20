import numpy as np
import sklearn as skl
from numpy.linalg import *


class DiscriminativeKalmanFilter(skl.base.BaseEstimator):
    """
    Implements the Discriminative Kalman Filter as described in Burkhart, M.C.,
    Brandman, D.M., Franco, B., Hochberg, L.R., & Harrison, M.T.'s "The
    discriminative Kalman filter for Bayesian filtering with nonlinear and
    nongaussian observation models." Neural Comput. 32(5), 969–1017 (2020).
    """

    def __init__(
        self,
        Α=None,
        Γ=None,
        S=None,
        f=None,
        Q=None,
        μₜ=None,
        Σₜ=None,
    ):
        self.Α = Α  # from eq. (2.1b)
        self.Γ = Γ  # from eq. (2.1b)
        self.S = S  # from eq. (2.1a)
        self.f = lambda x: f(x)  # from eq. (2.2)
        self.Q = lambda x: Q(x)  # from eq. (2.2)
        self.μₜ = μₜ  # from eq. (2.6)
        self.Σₜ = Σₜ  # from eq. (2.6)
        self.d = Α.shape[0]  # as in eq. (2)

    def stateUpdate(self):
        """
        calculates the first 2 lines of eq. (2.7) in-place
        """
        self.μₜ = self.Α @ self.μₜ
        self.Σₜ = self.Α @ self.Σₜ @ self.Α.T + self.Γ

    def measurementUpdate(self, newMeasurement):
        """
        calculates the last 2 lines of eq. (2.7)
        :param newMeasurement: new observation
        """
        Qxₜ = self.Q(newMeasurement)
        fxₜ = self.f(newMeasurement)
        if not np.all(eigvals(inv(Qxₜ) - inv(self.S)) > 1e-6):
            Qxₜ = inv(inv(Qxₜ) + inv(self.S))
        newPosteriorCovInv = inv(self.Σₜ) + inv(Qxₜ) - inv(self.S)
        self.μₜ = np.mat(
            solve(
                newPosteriorCovInv,
                solve(self.Σₜ, self.μₜ) + solve(Qxₜ, fxₜ),
            )
        )
        self.Σₜ = inv(newPosteriorCovInv)

    def predict(self, newMeasurement):
        """
        performs stateUpdate() and measurementUpdate(newMeasurement)
        :param newMeasurement: new observation
        :return: new posterior mean as in eq. (2.7)
        """
        self.stateUpdate()
        self.measurementUpdate(newMeasurement)
        return self.μₜ
