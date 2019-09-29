abstract type RegressionModels end
#If only we could add methods to abstract types...
# (M::RegressionModel)(X) = RegressionOut(X, M)
# maybe in Julia 2.0? - Update: Julia 1.2 allows this!!!

struct ClassicLeastSquares <: RegressionModels
    Coefficients::Array
    Bias::Bool
end

struct RidgeRegression <: RegressionModels
    CLS::ClassicLeastSquares
end

"""
    ClassicLeastSquares( X, Y; Bias = false )

Constructs a ClassicLeastSquares regression model of the form `Y` = A`X` with or without a `Bias` term. Returns a CLS object.
"""
function ClassicLeastSquares( X, Y; Bias = false )
    Z = (Bias) ? hcat( ones( size( X )[ 1 ] ), X ) : X
    return ClassicLeastSquares(Base.inv(Z' * Z) * Z' * Y, Bias)
end

"""
    OrdinaryLeastSquares( X, Y; Bias = false )

Makes a ClassicLeastSquares regression model of the form `Y` = A`X` with or without a `Bias` term. Returns a CLS object.
This is a wrapper function for CLS, because most other fields refer to this as OLS.
"""
function OrdinaryLeastSquares( X, Y; Bias = false )
    Z = (Bias) ? hcat( ones( size( X )[ 1 ] ), X ) : X
    return ClassicLeastSquares(Base.inv(Z' * Z) * Z' * Y, Bias)
end

function PredictFn(X, M::Union{ClassicLeastSquares, RidgeRegression})
    Z = ( M.Bias ) ? hcat( ones(size( X )[ 1 ] ), X ) : X
    return Z * M.Coefficients
end
"""
    (M::ClassicLeastSquares)(X)

Makes an inference from `X` using a ClassicLeastSquares object.
"""
(M::ClassicLeastSquares)(X) = PredictFn(X, M)

"""
    RidgeRegression( X, Y, Penalty; Bias = false )

Makes a RidgeRegression model of the form `Y` = A`X` with or without a `Bias` term and has an L2 `Penalty`. Returns a CLS object.
"""
function RidgeRegression( X, Y, Penalty; Bias = false )
    Y = forceMatrix(Y)
    Z = (Bias) ? hcat( ones( size( X )[ 1 ] ), X ) : X
    return RidgeRegression( ClassicLeastSquares( Base.inv( (Z' * Z) .+ (Penalty .* Diagonal( ones( size(Z)[2] ) ) ) ) * Z' * Y, Bias) )
end

"""
    (M::RidgeRegression)(X)

Makes an inference from `X` using a RidgeRegression object which wraps a ClassicLeastSquares object.
"""
(M::RidgeRegression)(X) = PredictFn(X, M.CLS)

struct KRR
    kernel::Kernel
    RR::RidgeRegression
end

"""
    KernelRidgeRegression( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )

Makes a KernelRidgeRegression model of the form `Y` = A`K` using a user specified Kernel("Linear", or "Guassian") and has an L2 `Penalty`.
Returns a KRR Wrapper for a CLS object.
"""
function KernelRidgeRegression( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )
    Kern = Kernel( KernelParameter, KernelType, X )
    return KRR(Kern, RidgeRegression( Kern(X), Y, Penalty ) )
end
"""
    (M::KRR)(X)

Makes an inference from `X` using a KRR object which wraps a ClassicLeastSquares object.
"""
(M::KRR)(X) = PredictFn(M.kernel(X), M.RR.CLS)

struct LSSVM
    kernel::Kernel
    RR::RidgeRegression
end

function formatlssvminput(X)
    Y = zeros(size(X) .+ 1)
    Y[ 1, 1 ] = 1.0
    Y[ 2:end, 2:end ] .= X
    return Y
end

"""
    LSSVM( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )

Makes a LSSVM model of the form `Y` = A`K` with a bias term using a user specified Kernel("linear", or "gaussian") and has an L2 `Penalty`.
Returns a LSSVM Wrapper for a CLS object.
"""
function LSSVM( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )
    Kern = Kernel( KernelParameter, KernelType, X )
    return LSSVM(Kern, RidgeRegression( formatlssvminput( Kern( X ) ), vcat( 0.0, Y ), Penalty ) )
end

"""
    (M::LSSVM)(X)

Makes an inference from `X` using a LSSVM object.
"""
(M::LSSVM)(X) = PredictFn( formatlssvminput( M.kernel(X)), M.RR.CLS )[2:end,:]

struct PrincipalComponentRegression <: RegressionModels
    PCA::PCA
    CLS::ClassicLeastSquares
end

"""
    PrincipalComponentRegression(PCAObject, Y )

Makes a PrincipalComponentRegression model object from a PCA Object and property value `Y`.
"""
function PrincipalComponentRegression(PCAObject::PCA, Y )
    return PrincipalComponentRegression(PCAObject, ClassicLeastSquares( PCAObject.Scores, forceMatrix(Y) ) )
end

PredictFn(X, M::PrincipalComponentRegression) = M.CLS( M.PCA( X ) )
"""
    (M::PrincipalComponentRegression)( X )

Makes an inference from `X` using a PrincipalComponentRegression object.
"""
(M::PrincipalComponentRegression)( X ) = PredictFn( X, M )

struct PartialLeastSquares <: RegressionModels
    XLoadings::Array
    XScores::Array
    YLoadings::Array
    YScores::Array
    XWeights::Array
    RWeights::Array
    Coefficients::Array
    Factors::Int
    XVariance::Array
    YVariance::Array
end

"""
    PartialLeastSquares( X, Y; Factors = minimum(size(X)) - 2, tolerance = 1e-8, maxiters = 200 )

Returns a PartialLeastSquares regression model object from arrays `X` and `Y`.

1. PARTIAL LEAST-SQUARES REGRESSION: A TUTORIAL PAUL GELADI and BRUCE R.KOWALSKI. Analytica Chimica Acta, 186, (1986) PARTIAL LEAST-SQUARES REGRESSION:
2. Martens H., NÊs T. Multivariate Calibration. Wiley: New York, 1989.
3. Re-interpretation of NIPALS results solves PLSR inconsistency problem. Rolf Ergon. Published in Journal of Chemometrics 2009; Vol. 23/1: 72-75
"""
function PartialLeastSquares( X, Y; Factors = minimum(size(X)) - 2, tolerance = 1e-8, maxiters = 200 )
    (Xrows, Xcols) = size(X)
    Y = forceMatrix(Y)
    (Yrows, Ycols) = size(Y)
    @assert Factors < (reduce(min, (Xrows, Xcols) ) - 1)
    Xd = copy(X); Yd = copy(Y)
    Xvar = Statistics.var(X, dims = 1)
    Yvar = Statistics.var(Y, dims = 1)
    Coefficients = []
    T = zeros(Xrows, Factors); t = zeros(Xrows); tprime = zeros(Xrows);
    U = zeros(Yrows, Factors); u = zeros(Yrows)
    P = zeros(Xcols, Factors); p = zeros(Xcols);
    Q = zeros(Ycols, Factors); q = zeros(Ycols)
    W = zeros(Xcols, Factors); w = zeros(Xrows);
    for factor in 1:Factors
        u = (Ycols == 1) ? Yd[:,1] : Yd[ :, argmax( [ (Yd[:,col]' * Yd[:,col])[1] for col in 1:Ycols] ) ]
        for iter in 1:maxiters
            w = X' * u
            w = w ./ sqrt.(w' * w)
            t = Xd * w
            q = Y' * (t ./ (t' * t) )
            diff = t .- tprime
            if (Ycols == 1) || ( sqrt.( diff' * diff )[1] < tolerance )
                break
            end
            u = Y * (q ./ (q' * q))
            tprime = t;
        end#End for ALS iterations
        tnorm = t ./ ( t' * t )
        p = X' * tnorm
        Xd = Xd .- ( t * p' )
        Yd = Yd .- ( t * q' )
        #Update Model Variables
        Q[:,factor] = q; T[:,factor] = t
        U[:,factor] = u; P[:,factor] = p
        W[:, factor] = w
    end#end for factors
    R = (Factors == 1) ? W : W * Base.inv( P' * W )
    #Use a more modern way to solve for the regression coefficients (2)
    Coefficients = R * Q'
    #An Equivalent way to obtain the regression coefficients.
    #Coefficients = W*Base.inv(W'*X'*X*W)*W'*X'*Y*Q
    return PartialLeastSquares(P, T, Q, U, W, R, Coefficients, Factors, Xvar, Yvar)
end

function PredictFn(X, M::PartialLeastSquares; Factors)
    Coeffs = []
    if Factors < M.Factors
        R = (Factors == 1) ? M.XWeights[:, 1:Factors] : M.XWeights[:, 1:Factors] * Base.inv( M.XLoadings[:, 1:Factors]' * M.XWeights[:, 1:Factors]  )
        Coeffs = R * M.YLoadings[:, 1:Factors]'
    else
        Coeffs = M.Coefficients
    end
    return X * Coeffs
end

"""
    (M::PartialLeastSquares)

Makes an inference from `X` using a PartialLeastSquares object.
"""
(M::PartialLeastSquares)(X; Factors = M.Factors) = PredictFn(X, M; Factors = Factors)

struct ELM <: RegressionModels
    Reservoir::Array
    Coefficients::Array
    Fn::Function
end

"""
    sigmoid(x)

Applies the sigmoid function to a scalar value X. Returns a scalar. Can be broad-casted over an Array.
"""
sigmoid(x) = 1.0 / (1.0 + exp(-1.0 * x))

"""
    ExtremeLearningMachine(X, Y, ReservoirSize = 10; ActivationFn = sigmoid)

Returns a ELM regression model object from arrays `X` and `Y`, with a user specified `ReservoirSize` and `ActivationFn`.

Extreme learning machine: a new learning scheme of feedforward neural networks. Guang-Bin Huang ; Qin-Yu Zhu ; Chee-Kheong Siew. 	2004 IEEE International Joint...
"""
function ExtremeLearningMachine(X, Y, ReservoirSize = 10; ActivationFn = sigmoid)
    W = randn( size( X )[2], ReservoirSize )
    return ELM( W, LinearAlgebra.pinv( ActivationFn.(X * W) ) * Y,
                ActivationFn)
end

PredictFn(X, M::ELM) = M.Fn.(X * M.Reservoir) * M.Coefficients;
"""
    (M::ELM)(X)

Makes an inference from `X` using a ELM object.
"""
(M::ELM)(X) = PredictFn(X, M)

mutable struct block
    value::Float64 #Avg value
    weight::Float64
    size::Int32
    previous::Int32
    next::Int32
end

"""
    MonotoneRegression(x, w = nothing)

Performs a monotone/isotonic regression on a vector x. This can be weighted
with a vector w.

Exceedingly Simple Monotone Regression. Jan de Leeuw. Version 02, March 30, 2017
"""
function MonotoneRegression(x::Array{Float64,1}, w = nothing)
    bins = length(x)
    retx = ones(bins)
    retx[:] = x[:]
    if isa(w, Nothing)
        w = ones(bins)
    end
    bins = bins + 1
    blocks = [ block(x[i], w[i], 1, i-1, i+1) for i in 1:(bins-1)]
    #I have a one off error somewhere an easy solution is just to pad
    push!(blocks, block( 0.0, w[end], 1, bins-1, bins+1) )
    active = 1
    continue_reg = true
    while( continue_reg )
        upsatisfied = false
        next = blocks[active].next
        if next == bins
            upsatisfied = true
        elseif blocks[next].value > blocks[active].value
            upsatisfied = true
        end
        if !upsatisfied
            ww = blocks[active].weight + blocks[next].weight
            nextnext = blocks[next].next
            wxactive = blocks[active].weight * blocks[active].value
            wxnext = blocks[next].weight * blocks[next].value
            blocks[active].value = (wxactive + wxnext) / ww
            blocks[active].weight = ww
            blocks[active].size += blocks[next].size
            blocks[active].next = nextnext
            if nextnext < bins
                blocks[nextnext].previous = active
            end
            blocks[next].size = 0
        end
        downsatisfied = false
        previous = blocks[active].previous

        if previous == 0
            downsatisfied = true
        elseif blocks[previous].value < blocks[active].value
            downsatisfied = true
        end
        if !downsatisfied
            ww = blocks[active].weight + blocks[previous].weight
            previousprevious = blocks[previous].previous
            wxactive = blocks[active].weight * blocks[active].value
            wxprevious = blocks[previous].weight * blocks[previous].value
            blocks[active].value = (wxactive + wxprevious) / ww
            blocks[active].weight = ww
            blocks[active].size += blocks[previous].size
            blocks[active].previous = previousprevious
            if previousprevious > 0
                blocks[previousprevious].next = active
            end
            blocks[previous].size = 0
        end
        if (blocks[active].next == bins) && (downsatisfied)
            continue_reg = false
        end
        if (upsatisfied && downsatisfied)
            active = next
        end
    end
    k = 1
    for i in 1:bins
        blksize = blocks[i].size;
        if (blksize > 0) && (i < bins )
            retx[k:(k+(blksize-1))] .= blocks[i].value;
            k += blksize
        end
    end
    return retx
end
