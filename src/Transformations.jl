using StatsBase
using LinearAlgebra

abstract type Transform end

struct Pipeline
    transforms
    inplace::Bool
end

#Naive Constructor...
Pipeline(Transforms) = Pipeline(Transforms, false)

function PipelineInPlace( X, FnStack...)
    pipeline = Array{Any,1}(undef, length(FnStack))
    for (i, fn) in enumerate( FnStack )
        pipeline[i] = fn(X)
        X .= pipeline[i]( X )
    end
    return Pipeline(Tuple(pipeline), true)
end

function Pipeline( X, FnStack...)
    pipeline = Array{Any,1}(undef, length(FnStack))
    for (i, fn) in enumerate( FnStack )
        pipeline[i] = isa(fn, Function) ? fn : fn(X)
        X = pipeline[i]( X )
    end
    return Pipeline(Tuple(pipeline), false)
end

function (P::Pipeline)(X; inverse = false)
    if inverse
        @assert any( isa.(P.transforms, Function) ) == false
        println(!any( isa.(P.transforms, Function) ))
        if P.inplace
            for fn in reverse( P.transforms ); X .= fn( X; inverse = true ) ; end
        else
            foldr( ( p, X ) -> p(X; inverse = true), P.transforms, init = X)
        end
    else
        if P.inplace
            for fn in enumerate( X.transforms ); X .= fn( X ) ; end
        else
            foldl( ( X, p ) -> p(X), P.transforms, init = X)
        end
    end
end


struct Center{B} <: Transform
    Mean::B
    invertible::Bool
end

Center(Z) = Center( StatsBase.mean(Z, dims = 1), true )

#Call with new data transforms the new data, or inverts it
(T::Center)(Z; inverse = false) = (inverse) ? (Z .+ T.Mean) : (Z .- T.Mean)

struct Scale{B} <: Transform
    StdDev::B
    invertible::Bool
end
Scale(Z) = Scale( StatsBase.std(Z, dims = 1),  true )
(T::Scale)(Z; inverse = false) = (inverse) ? (Z .* T.StdDev) : (Z ./ T.StdDev)

struct StandardNormalVariate{B,C} <: Transform
    Mean::B
    StdDev::C
end

function StandardNormalVariate(Z)
    mu = StatsBase.mean(Z, dims = 1)
    stdev = StatsBase.std(Z, dims = 1)
    StandardNormalVariate( mu, stdev)
end
#Call with new data transforms the new data, or inverts it
(T::StandardNormalVariate)(Z; inverse = false) = (inverse) ? ((Z .* T.StdDev) .+ T.Mean) : ((Z .- T.Mean) ./ T.StdDev)

struct RangeNorm{B,C} <: Transform
    Mins::B
    Maxes::C
end

function RangeNorm( Z )
    mins = minimum(Z, dims = 1)
    maxes = maximum(Z, dims = 1)
    RangeNorm( mins, maxes)
end

(T::RangeNorm)(Z; inverse = false) = (inverse) ? ( (Z .* ( T.Maxes .- T.Mins ) ) .+ T.Mins) : (Z .- T.Mins) ./ ( T.Maxes .- T.Mins )

struct MultiplicativeScatterCorrection <: Transform
    Bias
    Coefficients
end

function MultiplicativeScatterCorrection(Z)
    BiasedMeans = hcat( ones( ( size(Z)[2], 1) ) , StatsBase.mean( Z, dims = 1 )[1,:] )
    Coeffs = ( BiasedMeans' * BiasedMeans ) \ ( Z * BiasedMeans )'
    MultiplicativeScatterCorrection( Coeffs[1,:], Coeffs[2,:] )
end

(T::MultiplicativeScatterCorrection)(Z; inverse = false) = (inverse) ? ((Z .* T.Coefficients).+ T.Bias)  : (Z .- T.Bias) ./ T.Coefficients

struct PCA <: Transform
    Scores
    Loadings
    SingularValues
    algorithm::String
end

#NIPALS based PCA.
#Kind of advantageous is you don't want to outright compute all latent variables.
#Kind of slow, but a must for a chemometrics package...
function PCA_NIPALS(X; Factors = min(size(X)) - 1, tolerance = 1e-7, maxiters = 200)
    tolsq = tolerance * tolerance
    #Instantiate some variables up front for performance...
    Xsize = size(X)
    Tm = zeros( ( Xsize[1], Factors ) )
    Pm = zeros( ( Factors, Xsize[2] ) )
    t = zeros( ( 1, Xsize[1] ) )
    p = zeros( ( 1, Xsize[2] ) )
    #Set tolerance to floating point precision
    Residuals = copy(X)
    for factor in 1:Factors
        lastErr = sum(abs.(Residuals)); curErr = tolerance + 1;
        t = Residuals[:, 1]
        iterations = 0
        while (abs(curErr - lastErr) > tolsq) && (iterations < maxiters)
            p = Residuals' * t
            p = p ./ sqrt.( p' * p )
            t = Residuals * p
            #Track change in Frobenius norm
            lastErr = curErr
            curErr = sqrt(sum( ( Residuals - ( t * p' ) ) .^ 2))
            iterations += 1
        end
        Residuals -= t * p'
        Tm[:,factor] = t
        Pm[factor,:] = p
    end
    #Find singular values/eigenvalues
    EigVal = sqrt.( LinearAlgebra.diag( Tm' * Tm ) )
    #Scale loadings by singular values
    Tm = Tm * LinearAlgebra.Diagonal( 1.0 / EigVal )
    return PCA(Tm, Pm, EigVal, "NIPALS")
end

#SVD based PCA
function PCA(Z; Factors = min(size(Z)) - 1)
    svdres = LinearAlgebra.svd(Z)
    return PCA(svdres.U[:, 1:Factors], svdres.Vt[1:Factors, :], svdres.S[1:Factors], "SVD")
end

#Calling a PCA object on new data brings the new data into the PCA transforms basis...
(T::PCA)(Z::Array; Factors = 1, inverse = false) = (inverse) ? Z * (Diagonal(T.SingularValues[1:Factors]) * T.Loadings[:,1:Factors]) : Z * (Diagonal( 1 ./ T.SingularValues[1:Factors]) * T.Loadings[:,1:Factors])'
(T::PCA)(Z::Array; inverse = false) = (inverse) ? Z * (Diagonal(T.SingularValues) * T.Loadings) : Z * (Diagonal( 1 ./ T.SingularValues) * T.Loadings)'

ExplainedVariance(PCA::PCA) = ( PCA.SingularValues .^ 2 ) ./ sum( PCA.SingularValues .^ 2 )
