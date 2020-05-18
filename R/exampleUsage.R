source("DiscriminativeKalmanFilter.R")

library(R.matlab)
library(MASS)
library(ranger)

set.seed(0)

# data source
procd <- readMat("../data/exampleData.mat")
vel <- matrix(unlist(procd$procd[1]), ncol = 2)
spk <- matrix(unlist(procd$procd[2]), ncol = 10)
z <- vel[2:nrow(vel), ]
x <- spk[1:nrow(vel) - 1, ]

# dimensions of latent states and observations, respectively
dz <- ncol(z)
dx <- ncol(x)

# training data
train_idx <- 1:5000
n_train <- length(train_idx)
x_train <- x[train_idx, ]
z_train <- z[train_idx, ]

# test data
test_idx <- 5001:6000
n_test <- length(test_idx)
x_test <- data.frame(x[test_idx, ])
z_test <- z[test_idx, ]

# learn state model parameters A & Gamma from Eq. (2.1b)
st_mdl <- lm(z_train[2:n_train, ] ~ z_train[1:n_train - 1, ] - 1)
A0 <- t(st_mdl$coefficients)
Gamma0 <- crossprod(st_mdl$residuals)/n_train

# split training set in order to train f() and Q() from Eq. (2.2) separately
train_mean_idx <- sample(seq_len(n_train), size = 0.9 * n_train)
train_covariance_idx <- setdiff(seq_len(n_train), train_mean_idx)
x_train_mean <- x_train[train_mean_idx, ]
x_train_covariance <- data.frame(x_train[train_covariance_idx, ])
z_train_mean <- z_train[train_mean_idx, ]
z_train_covariance <- z_train[train_covariance_idx, ]

# learn f() as a random forest
df_train_mean1 <- data.frame(x_train_mean)
df_train_mean1["z1"] <- z_train_mean[, 1]
rf_mdl1 <- ranger(z1 ~ ., df_train_mean1, num.trees = 50)
df_train_mean2 <- data.frame(x_train_mean)
df_train_mean2["z2"] <- z_train_mean[, 2]
rf_mdl2 <- ranger(z2 ~ ., df_train_mean2, num.trees = 50)
fx <- function(x) t(as.matrix(cbind(predict(rf_mdl1, x)$predictions, predict(rf_mdl2, 
    x)$predictions)))

# learn Q() as a constant on held-out training data
z_train_preds <- as.matrix(cbind(predict(rf_mdl1, x_train_covariance)$predictions, 
    predict(rf_mdl2, x_train_covariance)$predictions))
cov_est <- crossprod(z_train_preds - z_train_covariance)/length(train_covariance_idx)
Qx <- function(x) cov_est

# initialize DKF using learned parameters
f0 <- fx(x_test[1, ])
Q0 <- Qx(x_test[1, ])
DKF <- DiscriminativeKalmanFilter$new(stateModelA = A0, stateModelGamma = Gamma0, 
    stateModelS = as.matrix(cov(z_train)), measurementModelF = fx, measurementModelQ = Qx, 
    currentPosteriorMean = f0, currentPosteriorCovariance = Q0)

# perform filtering
z_preds <- matrix(0, n_test, dz)
z_preds[1, ] <- f0
for (i in 2:length(test_idx)) {
    z_preds[i, ] <- DKF$predict(x_test[i, ])
}

# handle output
print("normalized rmse")
print(sqrt(mean((z_test - z_preds)^2))/sqrt(mean(z_test^2)))

