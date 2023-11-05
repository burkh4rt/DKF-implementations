import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split as skl_tt_split

from DiscriminativeKalmanFilter import DiscriminativeKalmanFilter

# data source
z = np.loadtxt("../data/z.csv", delimiter=",")
x = np.loadtxt("../data/x.csv", delimiter=",")

# dimensions of latent states and observations, respectively
dz, dx = z.shape[1], x.shape[1]

# training data
train_idx = range(5000)
n_train = len(train_idx)
x_train, z_train = (
    x[train_idx, :],
    z[train_idx, :],
)

# test data
test_idx = range(5000, 6000)
n_test = len(test_idx)
x_test, z_test = (
    x[test_idx, :],
    z[test_idx, :],
)

# learn state model parameters A & Gamma from Eq. (2.1b)
A0 = np.linalg.lstsq(z_train[1:, :], z_train[:-1, :], rcond=None)[0]
Gamma0 = np.mat(
    np.cov(
        z_train[1:, :] - z_train[:-1, :] @ A0,
        rowvar=False,
    )
)

# split training set in order to train f() and Q() from Eq. (2.2) separately
(
    x_train_mean,
    x_train_covariance,
    z_train_mean,
    z_train_covariance,
) = skl_tt_split(x_train, z_train, train_size=0.9)

# learn f() as a neural network
mean_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2),
    ]
)
mean_model.compile(optimizer="adam", loss="mean_squared_error")
mean_model.fit(x_train_mean, z_train_mean, epochs=25)
fx = lambda x: mean_model.predict(x.reshape(-1, dx))[0].reshape(dz, 1)

# learn Q() as a constant on held-out training data
z_train_preds = np.array(mean_model.predict_on_batch(x_train_covariance))
cov_est = np.zeros((dz, dz))
for i in range(z_train_preds.shape[0]):
    resid_i = np.mat(
        (z_train_preds[i, :] - z_train_covariance[i, :]).reshape(dz, 1)
    )
    cov_est += np.matmul(resid_i, resid_i.T) / z_train_preds.shape[0]
Qx = lambda x: cov_est

# initialize DKF using learned parameters
f0, Q0 = fx(x_test[0, :]), Qx(x_test[0, :])
DKF = DiscriminativeKalmanFilter(
    Α=A0,
    Γ=Gamma0,
    S=np.cov(z_train.T),
    f=fx,
    Q=Qx,
    μₜ=f0,
    Σₜ=Q0,
)

# perform filtering
z_preds = np.zeros_like(z_test)
z_preds[0, :] = f0.reshape(dz)
for i in range(1, n_test):
    z_preds[i, :] = DKF.predict(x_test[i, :]).flatten()

# handle output
print(
    "normalized rmse",
    np.sqrt(np.mean(np.square(z_test - z_preds)))
    / np.sqrt(np.mean(np.square(z_test))),
)
