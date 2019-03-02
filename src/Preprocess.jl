StandardNormalVariate(X) = ( X .- Statistics.mean(X, dims = 2) ) ./ Statistics.std(X, dims = 2)

Scale1Norm(X) = X ./ sum(abs.(X), dims = 2)
Scale2Norm(X) = X ./ sqrt.(sum(X .^ 2, dims = 2))
ScaleInfNorm(X) = X ./ reduce(max, X, dims = 2)
function ScaleMinMax(X)
    mini = reduce(min, X, dims = 2)
    maxi = reduce(max, X, dims = 2)
    return (X .- mini) ./ (maxi .- mini)
end

offsetToZero(X) = X .- reduce(min, X, dims = 2)

function boxcar(X; windowsize = 3, fn = mean)
    (obs, vars) = size(X)
    @assert windowsize <= vars
    result = zeros(obs, vars - windowsize + 1)
    for v in 1 : (vars - windowsize + 1 )
        result[:, v] = fn(X[:, v : (v + windowsize - 1) ], dims = 2)
    end
    return result
end


#Paul H. C. Eilers, Hans F.M. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing.  2005
function ALSSmoother(y; lambda = 100, p = 0.001, maxiters = 10)
    m = length(y)
    D = SecondDerivative( sparse( I, m, m )' )';
    w = ones(m);
    for it in 1 : maxiters
        W = spdiagm(0 => w);
        C = cholesky(W + lambda * D' * D).U;
        z = C \ (C' \ (w .* y));
        w[y .> z] .= p;
        w[y .< z] .+= 1.0 - p;
    end
    return z
end

#Paul H. C. Eilers. "A Perfect Smoother". Analytical Chemistry, 2003, 75 (14), pp 3631–3636.
function PerfectSmoother(y; lambda = 100, maxiters = 10)
    m = length(y)
    D = SecondDerivative( sparse( I, m, m )' )';
    w = spdiagm(0 => ones(m));
    C = cholesky(W + lambda * D' * D).U
    return C \ (C' \ (w * y))
end

#Pretty sure this is reversible like a transform, but don't have time to solve it
#in reverse yet...
struct MultiplicativeScatterCorrection
    BiasedMeans
    Bias
    Coefficients
end

function MultiplicativeScatterCorrection(Z)
    BiasedMeans = hcat( ones( ( size(Z)[2], 1) ) , StatsBase.mean( Z, dims = 1 )[1,:] )
    Coeffs = ( BiasedMeans' * BiasedMeans ) \ ( Z * BiasedMeans )'
    MultiplicativeScatterCorrection( BiasedMeans, Coeffs[1,:], Coeffs[2,:] )
end

function (T::MultiplicativeScatterCorrection)(Z)
    Coeffs = ( T.BiasedMeans' * T.BiasedMeans ) \ ( Z * T.BiasedMeans )'
    return (Z .- Coeffs[1,:]) ./ Coeffs[2,:]
end

function FirstDerivative(X)
    X = (length(size(X)) == 1) ? reshape( X, 1,length(a) ) : X
    Xsize = size(X)
    XNew = zeros( Xsize[ 1 ] , Xsize[ 2 ] - 1)
    for c in 1 : ( Xsize[ 2 ] - 1 )
        XNew[ :, c ] = X[ :, c + 1 ] .- X[ :, c ]
    end
    return XNew
end

function SecondDerivative(X)
    X = (length(size(X)) == 1) ? reshape( X, 1,length(a) ) : X
    Xsize = size(X)
    XNew = zeros( Xsize[ 1 ], Xsize[ 2 ] - 2 )
    for c in 1 : ( Xsize[ 2 ] - 2 )
        XNew[ :, c ] = (X[ :, c + 2 ] .- X[ :, c + 1 ]) - (X[ :, c + 1 ] .- X[ : , c ] )
    end
    return XNew
end

#Fractional Derivative via Grunwald Leitnikov algorithm,
#Useful for preprocessing data, and electrochemistry.
#Allows for nonuniform spacing in X...
function FractionalDerivative(Y, X = 1 : length(Y); Order = 0.5)
    (Obs,Vars) = size(Y)
    ddy = zeros(Obs,Vars-1)
    w = zeros(Vars)
    w[1] = 1.0
    for var in 2:Vars
        w[ var ] = w[ var - 1 ] * ( 1.0 - ( Order + 1.0 ) / ( var - 1.0 ) )
    end
    for var in 2:Vars
        h = (length(X) > 1) ? X[ var ] - X[ var - 1 ] : X
        for obs in 1 : Obs#This could definitely be broadcasted
            ddy[ obs, var-1 ] = w[ 1 : var ]' * Y[obs, var : -1: 1 ] ./ (h^Order)
        end
    end
    return ddy
end

#This is a little buggy, but its a start...
function SavitzkyGolay(X, Delta, PolyOrder, windowsize)
    @assert (windowsize % 2) == 1
    (Obs,Vars) = size(X)#length(Y
    windowspan = (windowsize - 1) / 2
    basis = ( ( -windowspan ) : windowspan ) .^ ( 0 : PolyOrder )'
    A = ( basis' * basis) \ basis'
    DeltaFac = factorial(Delta)
    output = DeltaFac * DSP.conv(X[1,:], A[Delta + 1,: ])'
    for r in 2:Obs
        output = vcat( output, DeltaFac * DSP.conv(X[r,:], A[Delta + 1,: ])' )
    end
    return output
end

#Direct Standardization Calibration Transfer Method
struct DirectStandardizationXform
    pca::PCA
    TransferMatrix::Array
end

function DirectStandardization(InstrumentX1, InstrumentX2; Factors = minimum(collect(size(InstrumentX1))) - 1)
    pcamodel = PCA(InstrumentX1; Factors = Factors)
    InstrumentX2_DS = pcamodel(InstrumentX2; Factors = Factors, inverse = false)
    TransferMatrix = LinearAlgebra.pinv(InstrumentX2_DS) * pcamodel.Scores[:, 1:Factors]
    return DirectStandardizationXform(pcamodel, TransferMatrix)
end

#Get the new samples in the Master instrument space...
function (DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.Values))
    #Transform data into PCA
    Into = DSX.pca(X; Factors = Factors)
    Bridge = Into * DSX.TransferMatrix[1:Factors,1:Factors]
    return DSX.pca(Bridge; Factors = Factors, inverse = true)
end

struct OrthogonalSignalCorrection
    Weights::Array
    Loadings::Array
    Residuals::Array
end

function OrthogonalSignalCorrection(X, Y; Factors = 1)
    (Obs,Vars) = size(X)
    W = zeros(Vars, Factors)
    Loadings = zeros( Vars, Factors)
    Residuals = copy(X)
    M = LinearAlgebra.Diagonal(ones(Vars)) .- ( X' * Y * Base.inv(Y' * X * X' * Y) * Y' * X )
    P = svd( X * M )
    for factor in 1 : Factors
        Projection = M * Residuals' * ( P.U[ : , factor ] ./ P.S[ factor ] )
        W[ : , factor ] = Projection / sqrt(Projection' * Projection)
        t = Residuals * Projection
        p = Residuals' * (t ./ ( t' * t ))
        Loadings[ : , factor ] = p
        Residuals = Residuals .- (t * p')
    end
    return OrthogonalSignalCorrection(W, Loadings, Residuals)
end


function (OSC::OrthogonalSignalCorrection)(Z; Factors = 2)
    X = copy(Z)
    for factor in 1 : Factors
        X .-= (X * OSC.Weights[ : , factor ] * OSC.Loadings[:, factor ]')
    end
    return X
end

#Correlation Alignment for Unsupervised Domain Adaptation. Baochen Sun, Jiashi Feng, Kate Saenko
#This transfers X1 to X2.
#https://arxiv.org/abs/1612.01939
struct CORAL
    coralmat::Array
end
function CORAL(X1, X2; lambda = 1.0)
    (Obs1, vars) = size(X1)
    (Obs2, vars) = size(X2)
    d = lambda .* LinearAlgebra.Diagonal(repeat([1], vars))
    c1 = (1.0 / Obs1) * (X1' * X1) .+ d
    c2 = (1.0 / Obs2) * (X2' * X2) .+ d
    CORALXfer = c1 ^ (-1/2) * c2 ^ (1/2)
    return CORAL(CORALXfer)
end
(C::CORAL)(Z) = Z * C.coralmat


struct TransferByOrthogonalProjection
    Factors::Int
    vars::Int
    pca::PCA
end

function TransferByOrthogonalProjection(X1, X2; Factors = 1)
    (Obs,Vars) = size(X1)
    OneToTwo = PCA(X1 .- X2; Factors = Factors)
    return TransferByOrthogonalProjection(Factors, Vars, OneToTwo)
end

function (TbOP::TransferByOrthogonalProjection)(X1; Factors = TbOP.Factors)
    return X1 * (LinearAlgebra.Diagonal( ones( TbOP.vars ) ) .- (TbOP.pca.Loadings[1:Factors,:]' * TbOP.pca.Loadings[1:Factors,:]))
end
