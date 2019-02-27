#Forces Array1's to Array2s of the same shape...
#forceMatrix(a) = (length(size(a)) == 1) ? reshape( a, length(a), 1 ) : a

abstract type RegressionModels end
#If only we could add methods to abstract types...
# (M::RegressionModel)(X) = RegressionOut(X, M)
# maybe in Julia 2.0?

struct ClassicLeastSquares <: RegressionModels
    Coefficients::Array
    Bias::Bool
end

struct RidgeRegression <: RegressionModels
    CLS::ClassicLeastSquares
end

function ClassicLeastSquares( X, Y; Bias = false )
    Z = (Bias) ? hcat( repeat( [ 1 ], size( X )[ 1 ] ), X ) : X
    return ClassicLeastSquares(Base.inv(Z' * Z) * Z' * Y, Bias)
end

function PredictFn(X, M::Union{ClassicLeastSquares, RidgeRegression})
    Z = ( M.Bias ) ? hcat( repeat( [ 1 ], size( X )[ 1 ] ), X ) : X
    return Z * M.Coefficients
end
(M::ClassicLeastSquares)(X) = PredictFn(X, M)

function RidgeRegression( X, Y, Penalty; Bias = false )
    Y = forceMatrix(Y)
    Z = (Bias) ? hcat( repeat( [ 1 ], size( X )[ 1 ] ), X ) : X
    return RidgeRegression( ClassicLeastSquares( Base.inv( (Z' * Z) .+ (Penalty .* Diagonal( ones( size(Z)[2] ) ) ) ) * Z' * Y, Bias) )
end

# Wrapper for CLS predict function...
(M::RidgeRegression)(X) = PredictFn(X, M.CLS)

struct KRR
    kernel::Kernel
    RR::RidgeRegression
end

function KernelRidgeRegression( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )
    Kern = Kernel( KernelParameter, KernelType, X )
    return KRR(Kern, RidgeRegression( Kern(X), Y, Penalty ) )
end
#Hahaha, wrap a kernel ridge object to ridge which wraps to CLS...
(M::KRR)(X) = PredictFn(M.kernel(X), M.RR.CLS)

struct LSSVM
    kernel::Kernel
    RR::RidgeRegression
end

function formatlssvminput(X)
    Y = zeros(size(X) .+ 1)
    Y[1,1] = 1.0
    Y[2:end,2:end] .= X
    return Y
end
#Literally the only difference between KRR and LSSVM is adding a bias term...
function LSSVM( X, Y, Penalty; KernelParameter = 0.0, KernelType = "linear" )
    Kern = Kernel( KernelParameter, KernelType, X )
    return LSSVM(Kern, RidgeRegression( formatlssvminput( Kern( X ) ), vcat( 0.0, Y ), Penalty ) )
end
#Hahaha, wrap a kernel ridge object to ridge which wraps to CLS...
(M::LSSVM)(X) = PredictFn( formatlssvminput( M.kernel(X)), M.RR.CLS )[2:end,:]


struct PrincipalComponentRegression <: RegressionModels
    PCA::PCA
    CLS::ClassicLeastSquares
end

function PrincipalComponentRegression(PCAObject, Y )
    return PrincipalComponentRegression(PCAObject, ClassicLeastSquares( PCAObject.Scores, forceMatrix(Y) ) )
end

PredictFn(X, M::PrincipalComponentRegression) = M.CLS( M.PCA( X ) )
(M::PrincipalComponentRegression)( X ) = PredictFn( X, M )

struct PartialLeastSquares <: RegressionModels
    XLoadings::Array
    XScores::Array
    YLoadings::Array
    YScores::Array
    XWeights::Array
    Coefficients::Array
    Factors::Int
end

#PLS-2 algorithm, this was decided because it is the most general...
function PartialLeastSquares( X, Y; Factors = minimum(size(X)) - 2, tolerance = 1e-8, maxiters = 200 )
    #A TUTORIAL PAUL GELADI and BRUCE R.KOWALSKI. Analytica Chimica Acta,
    #186, (1986) PARTIAL LEAST-SQUARES REGRESSION:
    (Xrows, Xcols) = size(X)
    Y = forceMatrix(Y)
    (Yrows, Ycols) = size(Y)
    @assert Factors < (reduce(min, (Xrows, Xcols) ) - 1)
    Xd = copy(X); Yd = copy(Y)
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
            if (Ycols == 1) || ( sqrt( diff' * diff ) < tolerance ) ; break; end
            u = Y * (q ./ (q' * q))
            tprime = t;
        end#End for ALS iterations
        tnorm = t ./ ( t' * t )
        p = X' * tnorm
        Xd = Xd .- ( t * p' )
        Yd = Yd .- ( t * q' )
        #Update Model Variables
        T[:,factor] = t; Q[:,factor] = q
        U[:,factor] = u; P[:,factor] = p
        W[:, factor] = w
    end#end for factors
    #Use a more mdodern way to solve for the regression coefficients
    #Martens H., NÊs T. Multivariate Calibration. Wiley: New York, 1989.
    Coefficients = (Factors == 1) ? W * Q' : W * Base.inv( P' * W ) * Q'
    #An Equivalent way to obtain the regression coefficients.
    #Re-interpretation of NIPALS results solves PLSR inconsistency problem. Rolf Ergon
    #Published in Journal of Chemometrics 2009; Vol. 23/1: 72-75
    #Coefficients = W*Base.inv(W'*X'*X*W)*W'*X'*Y*Q
    return PartialLeastSquares(P, T, Q, U, W, Coefficients, Factors)
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
(M::PartialLeastSquares)(X; Factors = M.Factors) = PredictFn(X, M; Factors = Factors)

struct ELM <: RegressionModels
    Reservoir::Array
    Coefficients::Array
    Fn::Function
end

sigmoid(x) = 1.0 / (1.0 + exp(-1.0 * x))

function ExtremeLearningMachine(X, Y, ReservoirSize = 10; ActivationFn = sigmoid)
    W = randn( size( X )[2], ReservoirSize )
    return ELM( W, LinearAlgebra.pinv( ActivationFn.(X * W) ) * Y,
                ActivationFn)
end

PredictFn(X, M::ELM) = M.Fn.(X * M.Reservoir) * M.Coefficients;
(M::ELM)(X) = PredictFn(X, M)
