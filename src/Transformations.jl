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
