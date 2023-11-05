#!/usr/bin/env python3

import os
import jax

# Set backend env to JAX
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
import numpy as np
import jax.numpy as jnp

from sklearn.model_selection import train_test_split as skl_tt_split
from DiscriminativeKalmanFilter import DiscriminativeKalmanFilter, DKF_State

# data source
z = jnp.array(np.loadtxt("../data/z.csv", delimiter=","))
x = jnp.array(np.loadtxt("../data/x.csv", delimiter=","))

# dimensions of latent states and observations, respectively
dz, dx = z.shape[1], x.shape[1]

# training data
n_train = 5000
x_train, z_train = (
    x[:5000, :],
    z[:5000, :],
)

# test data
n_test = 1000
x_test, z_test = (
    x[5000:6000, :],
    z[5000:6000, :],
)

# learn state model parameters A & Gamma from Eq. (2.1b)
A0 = jnp.linalg.lstsq(z_train[1:, :], z_train[:-1, :], rcond=None)[0]
Gamma0 = jnp.cov(
    z_train[1:, :] - z_train[:-1, :] @ A0,
    rowvar=False,
)

# split training set in order to train f() and Q() from Eq. (2.2) separately
(
    x_train_mean,
    x_train_covariance,
    z_train_mean,
    z_train_covariance,
) = skl_tt_split(x_train, z_train, train_size=0.9)

# learn f() as a neural network
mean_model = keras.Sequential(
    [
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2),
    ]
)
mean_model.compile(optimizer="adam", loss="mean_squared_error")
mean_model.fit(x_train_mean, z_train_mean, epochs=25)
fx = lambda x: jnp.array(
    mean_model.predict(x.reshape(-1, dx))[0].reshape(dz, 1)
)

# learn Q() as a constant on held-out training data
z_train_preds = np.array(mean_model.predict_on_batch(x_train_covariance))
cov_est = jnp.zeros((dz, dz))
for i in range(z_train_preds.shape[0]):
    resid_i = (z_train_preds[i, :] - z_train_covariance[i, :]).reshape(dz, 1)
    cov_est += resid_i @ resid_i.T / z_train_preds.shape[0]
Qx = lambda x: cov_est

# initialize DKF using learned parameters
f0, Q0 = fx(x_test[0, :]), Qx(x_test[0, :])
state = DKF_State(f0, Q0)
DKF = DiscriminativeKalmanFilter(
    Α=A0,
    Γ=Gamma0,
    S=np.cov(z_train.T),
    f=fx,
    Q=Qx,
)

# perform filtering
z_preds = np.zeros_like(z_test)
z_preds[0, :] = f0.reshape(dz)
for i in range(1, n_test):
    pred, state = DKF.predict(state, x_test[i, :])
    z_preds[i, :] = pred.flatten()

# handle output
print(
    "normalized rmse",
    np.sqrt(np.mean(np.square(z_test - z_preds)))
    / np.sqrt(np.mean(np.square(z_test))),
)
