#!/usr/bin/env python3

import dataclasses
import hashlib
import datetime
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv as j_inv, solve as j_solve


class DKF_State(NamedTuple):
    μₜ: jnp.ndarray  # from eq. (2.6)
    Σₜ: jnp.ndarray  # from eq. (2.6)


@dataclasses.dataclass
class DiscriminativeKalmanFilter:
    """
    Implements the Discriminative Kalman Filter as described in Burkhart, M.C.,
    Brandman, D.M., Franco, B., Hochberg, L.R., & Harrison, M.T.'s "The
    discriminative Kalman filter for Bayesian filtering with nonlinear and
    nongaussian observation models." Neural Comput. 32(5), 969–1017 (2020).
    """

    Α: jnp.ndarray  # from eq. (2.1b)
    Γ: jnp.ndarray  # from eq. (2.1b)
    S: jnp.ndarray  # from eq. (2.1a)
    f: callable  # from eq. (2.2)
    Q: callable  # from eq. (2.2)
    ts = datetime.datetime.now().isoformat()

    def __hash__(self):
        return hash(self.ts)

    @partial(jax.jit, static_argnames=["self"])
    def stateUpdate(self, state: DKF_State) -> DKF_State:
        """
        calculates the first 2 lines of eq. (2.7) in-place
        """
        μₜ = self.Α @ state.μₜ
        Σₜ = self.Α @ state.Σₜ @ self.Α.T + self.Γ
        return DKF_State(μₜ, Σₜ)

    @partial(jax.jit, static_argnames=["self"])
    def measurementUpdate(
        self, state: DKF_State, fxₜ: jnp.array, Qxₜ: jnp.array
    ) -> DKF_State:
        """
        calculates the last 2 lines of eq. (2.7)
        """
        Qxₜ = jax.lax.cond(
            jnp.any(jnp.linalg.eigvals(j_inv(Qxₜ) - j_inv(self.S)) > 1e-6),
            lambda: j_inv(j_inv(Qxₜ) + j_inv(self.S)),
            lambda: Qxₜ,
        )
        newPosteriorCovInv = j_inv(state.Σₜ) + j_inv(Qxₜ) - j_inv(self.S)
        μₜ = j_solve(
            newPosteriorCovInv,
            j_solve(state.Σₜ, state.μₜ) + j_solve(Qxₜ, fxₜ),
        )
        Σₜ = j_inv(newPosteriorCovInv)
        return DKF_State(μₜ, Σₜ)

    def predict(
        self, state: DKF_State, newMeasurement: jnp.ndarray
    ) -> tuple[jnp.ndarray, DKF_State]:
        """
        performs stateUpdate() and measurementUpdate(newMeasurement)
        :param newMeasurement: new observation
        :return: new posterior mean as in eq. (2.7)
        """
        Qxₜ = self.Q(newMeasurement)
        fxₜ = self.f(newMeasurement)
        state = self.stateUpdate(state)
        state = self.measurementUpdate(state, fxₜ, Qxₜ)
        return state.μₜ, state
