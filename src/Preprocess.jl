"""
    Noise(data, level; scale = :percent_max, type = :gaussian )

Adds noise (uniform, or gaussian) to an array of data.
The scale can be made relative to the percent_max, percent_average, or the absolute level values.

Note: for Gaussian noise it is likely convenient to scale by an inverse quantile(say 0.999)
"""
function Noise(data, level; scale = :percent_max, type = :gaussian )
    (maxes, minis) = ( [], [] )
    noise = []
    if type == :gaussian
        noise = randn( size( data ) )
    elseif type == :uniform
        noise = rand( Float64, size( data ) ) .- 0.5
    else #ToDo: Throw exception
        return
    end
    if scale == :percent_max
        if length( size( data ) ) > 1
            (maxes, minis) = ( reduce( max, data, dims = 2 ), reduce( min, data, dims = 2 ) )
            return ( noise .* (maxes .- minis) .* level ) .+ data
        else
            (maxes, minis) = ( reduce( max, data ), reduce( min, data ) )
            return ( noise .* (maxes .- minis) .* level ) .+ data
        end
    elseif scale == :percent_average
        if length(size(data)) > 1
            (maxes, minis) = ( reduce( max, data, dims = 2 ), reduce( min, data, dims = 2 ) )
            avg = mean( (data .- minis) ./ (maxes .- minis), dims = 2 )
            return ( noise .* avg .* level ) .+ data
        else
            (maxes, minis) = ( reduce( max, data ), reduce( min, data ) )
            avg = mean( (data .- minis) ./ (maxes .- minis) )
            return ( noise .* avg .* level ) .+ data
        end
    elseif scale == :absolute
        return ( noise .* level ) .+ data
    end
end

"""
    StandardNormalVariate(X)

Scales the columns of `X` by the mean and standard deviation of each row. Returns the scaled array.
"""
StandardNormalVariate(X) = ( X .- Statistics.mean(X, dims = 2) ) ./ Statistics.std(X, dims = 2)

"""
    Scale1Norm(X)

Scales the columns of `X` by the 1-Norm of each row. Returns the scaled array.
"""
Scale1Norm(X) = X ./ sum(abs.(X), dims = 2)

"""
    Scale2Norm(X)

Scales the columns of `X` by the 2-Norm of each row. Returns the scaled array.
"""
Scale2Norm(X) = X ./ sqrt.(sum(X .^ 2, dims = 2))

"""
    ScaleInfNorm(X)

Scales the columns of `X` by the Inf-Norm of each row. Returns the scaled array.
"""
ScaleInfNorm(X) = X ./ reduce(max, X, dims = 2)

"""
    ScaleFNorm(X)

Scales EACH entry of `X` by the Frobenius Norm of. Returns the scaled array.
"""
ScaleFNorm(X) = X ./ FNorm( X )

"""
    ScaleMinMax(X)

Scales the columns of `X` by the Min and Max of each row such that no observation is greater than 1 or less than zero.
Returns the scaled array.
"""
function ScaleMinMax(X)
    mini = reduce(min, X, dims = 2)
    maxi = reduce(max, X, dims = 2)
    return (X .- mini) ./ (maxi .- mini)
end

"""
    ScaleByIntensity(X, index)

Scales the columns of `X` by the value of a peak at a specified `index` in each row.
Returns the scaled array.
"""
ScaleByIntensity(X, index) = X ./ X[:,index]

"""
    offsetToZero(X)

Ensures that no observation(row) of Array `X` is less than zero, by ensuring the minimum value of each row is zero.
"""
offsetToZero(X) = X .- reduce(min, X, dims = 2)

"""
    boxcar(X; windowsize = 3, fn = mean)

Applies a boxcar function (`fn`) to each window of size `windowsize` to every row in `X`.
*Note: the function provided must support a dims argument/parameter.*
"""
function boxcar(X; windowsize = 3, fn = mean)
    X = forceMatrixT(X)
    (obs, vars) = size(X)
    @assert windowsize <= vars
    result = zeros(obs, vars - windowsize + 1)
    for v in 1 : (vars - windowsize + 1 )
        result[:, v] = fn(X[:, v : (v + windowsize - 1) ], dims = 2)
    end
    return result
end

"""
    ALSSmoother(X; lambda = 100, p = 0.001, maxiters = 10)

Applies an assymetric least squares smoothing function to a 2-Array `X`. The `lambda`, `p`, and `maxiters`
parameters control the smoothness. See the reference below for more information.

Paul H. C. Eilers, Hans F.M. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing.  2005
"""
function ALSSmoother(X; lambda = 100, p = 0.001, maxiters::Int = 10)
    X = forceMatrixT(X)
    (obs, vars) = size(X)
    output = zeros(obs,vars)
    for r in 1:obs
        D = SecondDerivative( sparse( I, vars, vars )' )';
        w = ones(vars);
        z = zeros(vars)
        for it in 1 : maxiters
            W = spdiagm(0 => w);
            C = cholesky(W + lambda * D' * D).U;
            z = C \ (C' \ (w .* X[r,:]));
            w[ X[r,:] .> z ] .= p;
            w[ X[r,:] .< z ] .+= 1.0 - p;
        end
        output[r,:] = z
    end
    return output
end

"""
    PerfectSmoother(X; lambda = 100)

Applies an assymetric least squares smoothing function to a a 2-Array `X`. The `lambda`
parameter controls the smoothness. See the reference below for more information.

Paul H. C. Eilers. "A Perfect Smoother". Analytical Chemistry, 2003, 75 (14), pp 3631–3636.
"""
function PerfectSmoother(X; lambda = 100)
    X = forceMatrixT(X)
    (obs, vars) = size(X)
    output = zeros(obs,vars)
    for r in 1:obs
        D = SecondDerivative( sparse( I, vars, vars )' )';
        w = spdiagm(0 => ones(vars));
        C = cholesky(w + lambda * D' * D).U
        output[r,:] = C \ (C' \ (w * X[r,:]))
    end
    return output
end

#Pretty sure this is reversible like a transform, but don't have time to solve the inverse yet...
struct MultiplicativeScatterCorrection
    BiasedMeans
    Bias
    Coefficients
end

"""
    MultiplicativeScatterCorrection(Z::Array)

Creates a MultiplicativeScatterCorrection object from the data in Z

Martens, H. Multivariate calibration. Wiley
"""
function MultiplicativeScatterCorrection(Z::Array)
    BiasedMeans = hcat( ones( ( size(Z)[2], 1) ) , StatsBase.mean( Z, dims = 1 )[1,:] )
    Coeffs = ( BiasedMeans' * BiasedMeans ) \ ( Z * BiasedMeans )'
    MultiplicativeScatterCorrection( BiasedMeans, Coeffs[1,:], Coeffs[2,:] )
end

"""
    (T::MultiplicativeScatterCorrection)(Z)

Applies MultiplicativeScatterCorrection from a stored object `T` to Array `Z`.
"""
function (T::MultiplicativeScatterCorrection)(Z)
    Coeffs = ( T.BiasedMeans' * T.BiasedMeans ) \ ( Z * T.BiasedMeans )'
    return (Z .- Coeffs[1,:]) ./ Coeffs[2,:]
end

"""
    FirstDerivative(X)

Uses the finite difference method to compute the first derivative for every row in `X`.
*Note: This operation results in the loss of a column dimension.*
"""
function FirstDerivative(X)
    X = forceMatrixT(X)
    Xsize = size(X)
    XNew = zeros( Xsize[ 1 ] , Xsize[ 2 ] - 1)
    for c in 1 : ( Xsize[ 2 ] - 1 )
        XNew[ :, c ] = X[ :, c + 1 ] .- X[ :, c ]
    end
    return XNew
end

"""
    FirstDerivative(X)

Uses the finite difference method to compute the second derivative for every row in `X`.
*Note: This operation results in the loss of two columns.*
"""
function SecondDerivative(X)
    X = forceMatrixT(X)
    Xsize = size(X)
    XNew = zeros( Xsize[ 1 ], Xsize[ 2 ] - 2 )
    for c in 1 : ( Xsize[ 2 ] - 2 )
        XNew[ :, c ] = (X[ :, c + 2 ] .- X[ :, c + 1 ]) - (X[ :, c + 1 ] .- X[ : , c ] )
    end
    return XNew
end

"""
    FractionalDerivative(Y, X = 1 : length(Y); Order = 0.5)

Calculates the Grunwald-Leitnikov fractional order derivative on every row of Array Y.
Array `X` is a vector that has the spacing between column-wise entries in `Y`. `X` can be a scalar if that is constant (common in spectroscopy).
`Order` is the fractional order of the derivative.
*Note: This operation results in the loss of a column dimension.*

The Fractional Calculus, by Oldham, K.; and Spanier, J. Hardcover: 234 pages. Publisher: Academic Press, 1974. ISBN 0-12-525550-0
"""
function FractionalDerivative(Y, X = 1 : length(Y); Order = 0.5)
    Y = forceMatrixT(Y)
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

"""
    ConvFilter1DFFT(a, filter)

Performs a 1D convolution of vector `filter` onto `a` via the FFT definition of a
 convolution. This method implicitly zeropads and truncates the result.

Note: This isn't highly optimized for performance. It was made when DSP.jl broke,
and broke the ChemometricsTools.jl package.
"""
function ConvFilter1DFFT(a, filter)
    @assert(length(filter) <= length(a), "Length of filter should be less than the length of the vector." )
    NewSize = length(a) + length(filter) - 1;
    ADiff = Int(round((NewSize - length(a)) ))
    filterDiff = Int(round((NewSize - length(filter)) ))
    X = fft( vcat( a, zeros( ADiff ) ) )
    H = fft( vcat( filter, zeros( filterDiff ) ) )
    return real.(ifft( X .* H ))[ (ADiff+1):(end-ADiff) ]
end

"""
    SavitzkyGolay(X, Delta, PolyOrder, windowsize)

Performs SavitskyGolay smoothing across every row in an Array `X`.
The `window size` is the size of the convolution filter, `PolyOrder` is the order of the polynomial,
and `Delta` is the order of the derivative.

Savitzky, A.; Golay, M.J.E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures". Analytical Chemistry. 36 (8): 1627–39. doi:10.1021/ac60214a047.
"""
function SavitzkyGolay(X, DerivOrder, PolyOrder, windowsize::Int)
    @assert( (windowsize % 2) == 1, "Window size must be an odd number" )
    @assert( PolyOrder < (windowsize-1), "Polynomial order must be less than the window size." )
    X = forceMatrixT(X)
    (Obs,Vars) = size(X)
    windowspan = (windowsize - 1) / 2
    basis = ones( windowsize, PolyOrder+1 )
    basis[:,2:end] = ( (-windowspan:windowspan)' .^ ( 1 : PolyOrder ))'
    DerivFac = factorial( DerivOrder )
    FIR = inv( basis' * basis) * basis'
    output = DerivFac * ConvFilter1DFFT(X[1,:], FIR[DerivOrder+1,: ])'
    for r in 2:Obs
        sg = DerivFac * ConvFilter1DFFT(X[r,:], FIR[DerivOrder+1,: ])'
        output = vcat( output, sg )
    end
    offset = Int( windowspan )
    return output[:, (offset + 1) : (end - offset)]
end

"""
    LinearResample(X, newsize)

Resamples a vector `X` to be of size `newsize` via linear interpolation.
"""
function LinearResample(X, newsize)
   cursize = length(X)
   if cursize == newsize
      return X
   end
   newspacing = ( (cursize-1) / (newsize-1) )
   newsampling = 1 : newspacing : cursize
   interpolated = zeros(newsize)
   for i in 1 : ( cursize - 1 )
      #For large arrays this is costly - could change to track the indices
      #assuming uniform spacing.
      inds = findall( i .<= newsampling .<= ( i + 1 ) )
      slope = ( X[i+1] - X[i] )
      offset = X[i] - (slope*i)
      interpolated[inds] = (newsampling[inds] .* slope) .+ offset
   end
   return interpolated
end

struct DirectStandardizationXform
    pca::PCA
    TransferMatrix::Array
end

"""
    DirectStandardization(InstrumentX1, InstrumentX2; Factors = minimum(collect(size(InstrumentX1))) - 1)

Makes a DirectStandardization object to facilitate the transfer from Instrument #2 to Instrument #1 .
The returned object can be used to transfer unseen data to the approximated space of instrument 1.
The number of `Factors` used are those from the internal orthogonal basis.

Yongdong Wang and Bruce R. Kowalski, "Calibration Transfer and Measurement Stability of Near-Infrared Spectrometers," Appl. Spectrosc. 46, 764-771 (1992)
"""
function DirectStandardization(InstrumentX1, InstrumentX2; Factors = minimum(collect(size(InstrumentX1))) - 1)
    pcamodel = PCA(InstrumentX1; Factors = Factors)
    InstrumentX2_DS = pcamodel(InstrumentX2; Factors = Factors, inverse = false)
    TransferMatrix = LinearAlgebra.pinv(InstrumentX2_DS) * pcamodel.Scores[:, 1:Factors]
    return DirectStandardizationXform(pcamodel, TransferMatrix)
end

"""
    (DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.Values))

Applies a the transform from a learned direct standardization object `DSX` to new data `X`.
"""
function (DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.Values))
    Into = DSX.pca(X; Factors = Factors)
    Bridge = Into * DSX.TransferMatrix[1:Factors,1:Factors]
    return DSX.pca(Bridge; Factors = Factors, inverse = true)
end

struct OrthogonalSignalCorrection
    Weights::Array
    Loadings::Array
    Residuals::Array
end

"""
    OrthogonalSignalCorrection(X, Y; Factors = 1)

Performs Thomas Fearn's Orthogonal Signal Correction to an endogenous `X` and exogenous `Y`.
The number of `Factors` are the number of orthogonal components to be removed from `X`.
This function returns an OSC object.

Tom Fearn. On orthogonal signal correction. Chemometrics and Intelligent Laboratory Systems. Volume 50, Issue 1, 2000, Pages 47-52.
"""
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

"""
    (OSC::OrthogonalSignalCorrection)(Z; Factors = 2)

Applies a the transform from a learned orthogonal signal correction object `OSC` to new data `Z`.
"""
function (OSC::OrthogonalSignalCorrection)(Z; Factors = 2)
    X = copy(Z)
    for factor in 1 : Factors
        X .-= (X * OSC.Weights[ : , factor ] * OSC.Loadings[:, factor ]')
    end
    return X
end

struct TransferByOrthogonalProjection
    Factors::Int
    vars::Int
    pca::PCA
end

"""
    TransferByOrthogonalProjection(X1, X2; Factors = 1)

Performs Thomas Fearns Transfer By Orthogonal Projection to facilitate transfer from `X1` to `X2`. Returns a TransferByOrthogonalProjection object.

Anne Andrew, Tom Fearn. Transfer by orthogonal projection: making near-infrared calibrations robust to between-instrument variation. Chemometrics and Intelligent Laboratory Systems. Volume 72, Issue 1, 2004, Pages 51-56,
"""
function TransferByOrthogonalProjection(X1, X2; Factors = 1)
    (Obs,Vars) = size(X1)
    OneToTwo = PCA(X1 .- X2; Factors = Factors)
    return TransferByOrthogonalProjection(Factors, Vars, OneToTwo)
end

"""
    (TbOP::TransferByOrthogonalProjection)(X1; Factors = TbOP.Factors)

Applies a the transform from a learned transfer by orthogonal projection object `TbOP` to new data `X1`.
"""
function (TbOP::TransferByOrthogonalProjection)(X1; Factors = TbOP.Factors)
    return X1 * (LinearAlgebra.Diagonal( ones( TbOP.vars ) ) .- (TbOP.pca.Loadings[1:Factors,:]' * TbOP.pca.Loadings[1:Factors,:]))
end

struct CORAL
    coralmat::Array
end
"""
    CORAL(X1, X2; lambda = 1.0)

Performs CORAL to facilitate covariance based transfer from `X1` to `X2` with regularization parameter `lambda`. Returns a CORAL object.

Correlation Alignment for Unsupervised Domain Adaptation. Baochen Sun, Jiashi Feng, Kate Saenko. https://arxiv.org/abs/1612.01939
"""
function CORAL(X1, X2; lambda = 1.0)
    (Obs1, vars) = size(X1)
    (Obs2, vars) = size(X2)
    d = lambda .* LinearAlgebra.Diagonal(repeat([1], vars))
    c1 = (1.0 / Obs1) * (X1' * X1) .+ d
    c2 = (1.0 / Obs2) * (X2' * X2) .+ d
    CORALXfer = c1 ^ (-1/2) * c2 ^ (1/2)
    return CORAL(CORALXfer)
end

"""
    (C::CORAL)(Z)

Applies a the transform from a learned `CORAL` object to new data `Z`.
"""
(C::CORAL)(Z) = Z * C.coralmat
