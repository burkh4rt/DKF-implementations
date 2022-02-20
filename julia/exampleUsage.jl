"""
To run:
```
julia --project=. exampleUsage.jl
```
"""

using DelimitedFiles
using Flux
using Random
using Statistics

include("DiscriminativeKalmanFilter.jl")

# setup
rng = MersenneTwister(42)

# data source
z = readdlm("../data/z.csv", ',')
x = readdlm("../data/x.csv", ',')

# dimensions of latent states and observations, respectively
dz = size(z)[2]
dx = size(x)[2]

# training data
train_idx = 1:5000
n_train = size(train_idx)[1]
x_train = x[train_idx, :]
z_train = z[train_idx, :]

# test data
test_idx = 5001:6000
n_test = size(test_idx)[1]
x_test = x[test_idx, :]
z_test = z[test_idx, :]

# learn state model parameters A & Gamma from Eq. (2.1b)
A0 = z_train[2:end, :] \ z_train[1:end-1, :]
Gamma0 = cov(z_train[2:end, :] - z_train[1:end-1, :] * A0)

# split training set in order to train f() and Q() from Eq. (2.2) separately
perm = randperm(MersenneTwister(42), n_train)
x_train_mean = x_train[perm[1:Int(0.9 * 5000)], :]
x_train_covariance = x_train[perm[Int(0.9 * 5000)+1:end], :]
z_train_mean = z_train[perm[1:Int(0.9 * 5000)], :]
z_train_covariance = z_train[perm[Int(0.9 * 5000)+1:end], :]

function NeuralNetwork()
    return Chain(
        Dense(dx, 10, relu),
        Dropout(0.1),
        Dense(10, dz)
    )
end

m = NeuralNetwork()
data = Flux.Data.DataLoader((x_train_mean', z_train_mean'), batchsize = 100, shuffle = true, rng = rng)
loss(x, y) = sum(Flux.Losses.mse(m(x), y))

for i in 1:20
    Flux.train!(loss, Flux.params(m), data, Flux.Optimise.ADAM())
end

Q = cov(z_train_covariance - m(x_train_covariance')')

fx(x) = m(x)
Qx(x) = Q

f0 = fx(x_test[1, :])
Q0 = Qx(x_test[1, :])
S = cov(z_train)

DKF = DiscriminativeKalmanFilter(A0, Gamma0, S, fx, Qx, f0, Q0, dz)

# perform filtering
z_preds = zeros(size(z_test))
z_preds[1, :] = f0
for i in 2:n_test
    z_preds[i, :] = predict!(DKF, x_test[i, :])
end

print("normalized rmse ", sqrt(mean((z_test - z_preds) .^ 2)) / sqrt(mean(z_test .^ 2)))
