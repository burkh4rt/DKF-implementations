
DiscriminativeKalmanFilter <- setRefClass("DiscriminativeKalmanFilter", fields = list(stateModelA = "matrix", 
    stateModelGamma = "matrix", stateModelS = "matrix", measurementModelF = "function", 
    measurementModelQ = "function", currentPosteriorMean = "matrix", currentPosteriorCovariance = "matrix", 
    dState = "integer"), methods = list(initialize = function(stateModelA, stateModelGamma, 
    stateModelS, measurementModelF, measurementModelQ, currentPosteriorMean, currentPosteriorCovariance) {
    dState <<- ncol(stateModelA)
    stateModelA <<- stateModelA
    stateModelGamma <<- stateModelGamma
    stateModelS <<- stateModelS
    measurementModelF <<- measurementModelF
    measurementModelQ <<- measurementModelQ
    currentPosteriorMean <<- currentPosteriorMean
    currentPosteriorCovariance <<- currentPosteriorCovariance
    stopifnot(dim(stateModelA) == c(dState, dState), dim(stateModelGamma) == c(dState, 
        dState), all(eigen(stateModelGamma)$values > 1e-06), isSymmetric(stateModelGamma), 
        dim(stateModelS) == c(dState, dState), all(eigen(stateModelS)$values > 1e-06), 
        isSymmetric(stateModelS), dim(currentPosteriorMean) == c(dState, 1), dim(currentPosteriorCovariance) == 
            c(dState, dState), all(eigen(currentPosteriorCovariance)$values > 1e-06), 
        isSymmetric(currentPosteriorCovariance))
}, stateUpdate = function() {
    currentPosteriorMean <<- stateModelA %*% currentPosteriorMean
    currentPosteriorCovariance <<- stateModelA %*% currentPosteriorCovariance %*% 
        t(stateModelA) + stateModelGamma
}, measurementUpdate = function(newMeasurement) {
    Qx <- measurementModelQ(newMeasurement)
    fx <- measurementModelF(newMeasurement)
    stopifnot(dim(fx) == c(dState, 1), dim(Qx) == c(dState, dState), all(eigen(Qx)$values > 
        1e-06), isSymmetric(Qx))
    if (!all(eigen(solve(Qx) - solve(stateModelS))$values > 1e-06)) {
        Qx <- solve(solve(Qx) + solve(stateModelS))
    }
    newPosteriorCovInv <- solve(currentPosteriorCovariance) + solve(Qx) - solve(stateModelS)
    currentPosteriorMean <<- solve(newPosteriorCovInv, solve(currentPosteriorCovariance, 
        currentPosteriorMean) + solve(Qx, fx))
    currentPosteriorCovariance <<- solve(newPosteriorCovInv)
}, predict = function(newMeasurement) {
    stateUpdate()
    measurementUpdate(newMeasurement)
    return(currentPosteriorMean)
}))



