
using LinearAlgebra

mutable struct DiscriminativeKalmanFilter
    A::Matrix{Float64}  # from eq. (2.1b)
    Γ::Matrix{Float64}  # from eq. (2.1b)
    S::Matrix{Float64}  # from eq. (2.1a)
    f::Function  # from eq. (2.2)
    Q::Function  # from eq. (2.2)
    μₜ::Vector{Float64}  # from eq. (2.6)
    Σₜ::Matrix{Float64}  # from eq. (2.6)
    d::Int  # as in eq. (2)
end

"""
calculates the first 2 lines of eq. (2.7) in-place
"""
function stateUpdate!(DKF::DiscriminativeKalmanFilter)
    DKF.μₜ = DKF.A * DKF.μₜ
    DKF.Σₜ = DKF.A * DKF.Σₜ * transpose(DKF.A) + DKF.Γ
end

"""
calculates the last 2 lines of eq. (2.7)
"""
function measurementUpdate!(DKF::DiscriminativeKalmanFilter, newMeasurement::Vector{Float64})
    Qxₜ = DKF.Q(newMeasurement)
    fxₜ = DKF.f(newMeasurement)
    if minimum(eigvals(inv(Qxₜ) - inv(DKF.S))) <= 1e-6
        newPosteriorCovInv = inv(DKF.Σₜ) + inv(Qxₜ)
    else
        newPosteriorCovInv = inv(DKF.Σₜ) + inv(Qxₜ) - inv(DKF.S)
    end
    DKF.μₜ = newPosteriorCovInv \ (DKF.Σₜ \ DKF.μₜ + Qxₜ \ fxₜ)
    DKF.Σₜ = inv(newPosteriorCovInv)
end

"""
performs stateUpdate() and measurementUpdate(newMeasurement);
returns new posterior mean as in eq. (2.7)
"""
function predict!(DKF::DiscriminativeKalmanFilter, newMeasurement::Vector{Float64})
    stateUpdate!(DKF)
    measurementUpdate!(DKF, newMeasurement)
    return DKF.μₜ
end
