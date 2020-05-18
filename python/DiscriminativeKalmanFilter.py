import numpy as np
import sklearn as skl
from numpy.linalg import *


class DiscriminativeKalmanFilter(skl.base.BaseEstimator):
    def __init__(
        self,
        stateModelA=None,
        stateModelGamma=None,
        stateModelS=None,
        measurementModelF=None,
        measurementModelQ=None,
        currentPosteriorMean=None,
        currentPosteriorCovariance=None,
    ):
        self.stateModelA = np.mat(stateModelA)
        self.stateModelGamma = np.mat(stateModelGamma)
        self.stateModelS = np.mat(stateModelS)
        self.measurementModelF = lambda x: np.mat(measurementModelF(x))
        self.measurementModelQ = lambda x: np.mat(measurementModelQ(x))
        self.currentPosteriorMean = np.mat(currentPosteriorMean)
        self.currentPosteriorCovariance = np.mat(currentPosteriorCovariance)
        self.dState = stateModelA.shape[0]
        assert self.stateModelA.shape == (self.dState, self.dState)
        assert self.stateModelGamma.shape == (self.dState, self.dState)
        assert np.all(eigvals(self.stateModelGamma) > 0.0)
        assert np.allclose(self.stateModelGamma, self.stateModelGamma.T)
        assert self.stateModelS.shape == (self.dState, self.dState)
        assert np.all(eigvals(self.stateModelS) > 0.0)
        assert np.allclose(self.stateModelS, self.stateModelS.T)
        assert self.currentPosteriorMean.shape == (self.dState, 1)
        assert self.currentPosteriorCovariance.shape == (self.dState, self.dState)
        assert np.all(eigvals(self.currentPosteriorCovariance) > 0.0)
        assert np.allclose(
            self.currentPosteriorCovariance, self.currentPosteriorCovariance.T
        )

    def stateUpdate(self):
        self.currentPosteriorMean = self.stateModelA * self.currentPosteriorMean
        self.currentPosteriorCovariance = (
            self.stateModelA * self.currentPosteriorCovariance * self.stateModelA.T
            + self.stateModelGamma
        )

    def measurementUpdate(self, newMeasurement):
        Qx = self.measurementModelQ(newMeasurement)
        fx = self.measurementModelF(newMeasurement)
        assert fx.shape == (self.dState, 1)
        assert Qx.shape == (self.dState, self.dState)
        assert np.all(eigvals(Qx) > 0.0)
        assert np.allclose(Qx, Qx.T)
        if not np.all(eigvals(inv(Qx) - inv(self.stateModelS)) > 1e-6):
            Qx = inv(inv(Qx) + inv(self.stateModelS))
        newPosteriorCovInv = np.mat(
            inv(self.currentPosteriorCovariance) + inv(Qx) - inv(self.stateModelS)
        )
        self.currentPosteriorMean = np.mat(
            solve(
                newPosteriorCovInv,
                solve(self.currentPosteriorCovariance, self.currentPosteriorMean)
                + solve(Qx, fx),
            )
        )
        self.currentPosteriorCovariance = np.mat(inv(newPosteriorCovInv))

    def predict(self, newMeasurement):
        self.stateUpdate()
        self.measurementUpdate(newMeasurement)
        return self.currentPosteriorMean
