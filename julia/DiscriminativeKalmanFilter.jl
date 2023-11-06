#!/usr/bin/env julia

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
    BLAS.gemv!('N', 1.0, DKF.A, DKF.μₜ, 0.0, DKF.μₜ) # DKF.μₜ = DKF.A * DKF.μₜ
    # DKF.Σₜ = DKF.A * DKF.Σₜ * transpose(DKF.A) + DKF.Γ
    BLAS.gemm!('N', 'T', 1.0, DKF.Σₜ, DKF.A, 0.0, DKF.Σₜ)
    BLAS.gemm!('N', 'N', 1.0, DKF.A, DKF.Σₜ, 0.0, DKF.Σₜ)
    BLAS.symm!('L', 'U', 1.0, Matrix((1.0 * I)(DKF.d)), DKF.Γ, 1.0, DKF.Σₜ)
end

"""
calculates the last 2 lines of eq. (2.7)
"""
function measurementUpdate!(
    DKF::DiscriminativeKalmanFilter,
    newMeasurement::Vector{Float64},
)
    Qxₜ = convert(Matrix{Float64}, DKF.Q(newMeasurement))
    fxₜ = convert(Vector{Float64}, DKF.f(newMeasurement))
    inv_Qxₜ = copy(Qxₜ)
    LAPACK.potrf!('U', inv_Qxₜ)
    LAPACK.potri!('U', inv_Qxₜ)
    inv_S = copy(DKF.S)
    LAPACK.potrf!('U', inv_S)
    LAPACK.potri!('U', inv_S)
    inv_Σₜ = copy(DKF.Σₜ)
    LAPACK.potrf!('U', inv_Σₜ)
    LAPACK.potri!('U', inv_Σₜ)
    if minimum(LAPACK.syev!('N', 'U', inv_Qxₜ - inv_S)) <= 1e-6
        # DKF.Σₜ = inv_Σₜ + inv_Qxₜ
        BLAS.gemm!('N', 'N', 1.0, inv_Σₜ, Matrix((1.0 * I)(DKF.d)), 0.0, DKF.Σₜ)
        BLAS.gemm!('N', 'N', 1.0, inv_Qxₜ, Matrix((1.0 * I)(DKF.d)), 1.0, DKF.Σₜ)
    else
        # DKF.Σₜ = inv_Σₜ + inv_Qxₜ - inv_S
        BLAS.gemm!('N', 'N', 1.0, inv_Σₜ, Matrix((1.0 * I)(DKF.d)), 0.0, DKF.Σₜ)
        BLAS.gemm!('N', 'N', 1.0, inv_Qxₜ, Matrix((1.0 * I)(DKF.d)), 1.0, DKF.Σₜ)
        BLAS.gemm!('N', 'N', -1.0, inv_S, Matrix((1.0 * I)(DKF.d)), 1.0, DKF.Σₜ)
    end
    LAPACK.potrf!('U', DKF.Σₜ)
    LAPACK.potri!('U', DKF.Σₜ)
    # DKF.μₜ = DKF.Σₜ * (inv_Σₜ * DKF.μₜ + inv_Qxₜ * fxₜ)
    BLAS.symv!('U', 1.0, inv_Σₜ, DKF.μₜ, 0.0, DKF.μₜ)
    BLAS.symv!('U', 1.0, inv_Qxₜ, fxₜ, 1.0, DKF.μₜ)
    DKF.μₜ = DKF.Σₜ * DKF.μₜ
    # BLAS.gemv!('N', 1., DKF.Σₜ, DKF.μₜ, 0., DKF.μₜ)
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
