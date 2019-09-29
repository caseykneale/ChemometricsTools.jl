abstract type Transform end

struct pipeline
    transforms
    inplace::Bool
end

#Naive Constructor...
"""
    Pipeline(Transforms)

Constructs a transformation pipeline from vector/tuple of `Transforms`. The Transforms vector are effectively a vector of functions which transform data.
"""
Pipeline(Transforms) = pipeline(Transforms, false)

"""
    PipelineInPlace( X, FnStack...)

Construct a pipeline object from vector/tuple of `Transforms`. The Transforms vector are effectively a vector of functions which transform data.
This function makes "inplace" changes to the Array `X` as though it has been sent through the pipeline.
This is more efficient if memory is a concern, but can irreversibly transform data in memory depending on the transforms in the pipeline.
"""
function PipelineInPlace( X, FnStack...)
    pipe = Array{Any,1}(undef, length(FnStack))
    for (i, fn) in enumerate( FnStack )
        pipe[i] = isa(fn, Function) ? fn : fn(X)
        X .= pipe[i]( X )
    end
    return pipeline(Tuple(pipe), true)
end

"""
    Pipeline( X, FnStack... )

Construct a pipeline object from vector/tuple of `Transforms`. The Transforms vector are effectively a vector of functions which transform data.
"""
function Pipeline( X, FnStack... )
    pipe = Array{Any,1}( undef, length( FnStack ))
    for (i, fn) in enumerate( FnStack )
        pipe[ i ] = isa( fn, Function ) ? fn : fn( X )
        X = pipe[ i ]( X )
    end
    return pipeline(Tuple( pipe ), false)
end

"""
    (P::pipeline)(X; inverse = false)

Applies the stored transformations in a pipeline object `P` to data in X.
The inverse flag can allow for the transformations to be reversed provided they are invertible functions.
"""
function (P::pipeline)(X; inverse = false)
    if inverse
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


struct QuantileTrim <: Transform
    Quantiles::Array
    invertible::Bool
end

"""
    QuantileTrim(Z; quantiles::Tuple{Float64,Float64} = (0.05, 0.95) )

Trims values above or below the specified columnwise quantiles to the quantile values themselves.
"""
function QuantileTrim(Z; quantiles::Tuple{Float64,Float64} = (0.05, 0.95) )
    @assert length(quantiles) == 2
    return QuantileTrim( EmpiricalQuantiles(Z, quantiles), false )
end

"""
    (T::QuantileTrim)(X, inverse = false)

Trims data in array `X` columns wise according to learned quantiles in QuantileTrim object `T`
This function does NOT have an inverse.
"""
function (T::QuantileTrim)(X, inverse = false)
    if inverse == false
        for c in 1:size(X)[2]
            if T.Quantiles[ 1, c ] != T.Quantiles[ 2, c ]
                lt = X[ : , c ] .< T.Quantiles[ 1, c ]
                gt = X[ : , c ] .> T.Quantiles[ 2, c ]
                X[ lt, c ] .= T.Quantiles[ 1, c ]
                X[ gt, c ] .= T.Quantiles[ 2, c ]
            end
        end
    else
        println("QuantileScale does not provide an inverse, skipping operation.")
    end
    return X
end

struct Center{B} <: Transform
    Mean::B
    invertible::Bool
end

"""
    Center(Z)

Acquires the mean of each column in `Z` provided and returns a transform that will subtract those column means from any future data.
"""
Center(Z) = Center( StatsBase.mean(Z, dims = 1), true )

"""
    (T::Center)(Z; inverse = false)

Centers data in array `Z` column-wise according to learned mean centers in Center object `T`.
"""
(T::Center)(Z; inverse = false) = (inverse) ? (Z .+ T.Mean) : (Z .- T.Mean)

struct Scale{B} <: Transform
    StdDev::B
    invertible::Bool
end

"""
    Scale(Z)

Acquires the standard deviation of each column in `Z` provided and returns a transform that will divide those column-wise standard deviation from any future data.
"""
Scale(Z) = Scale( StatsBase.std(Z, dims = 1),  true )

"""
    (T::Scale)(Z; inverse = false)

Scales data in array `Z` column-wise according to learned standard deviations in Scale object `T`.
"""
(T::Scale)(Z; inverse = false) = (inverse) ? (Z .* T.StdDev) : (Z ./ T.StdDev)

struct CenterScale{B,C} <: Transform
    Mean::B
    StdDev::C
    invertible::Bool
end

"""
    CenterScale(Z)

This is a composition of Center and Scale (in that order).
"""
function CenterScale(Z)
    mu = StatsBase.mean(Z, dims = 1)
    stdev = StatsBase.std(Z, dims = 1)
    CenterScale( mu, stdev, true)
end

"""
    (T::CenterScale)(Z; inverse = false)

Centers and Scales data in array `Z` column-wise according to learned measures of central tendancy in Scale object `T`.
"""
(T::CenterScale)(Z; inverse = false) = (inverse) ? ((Z .* T.StdDev) .+ T.Mean) : ((Z .- T.Mean) ./ T.StdDev)

struct RangeNorm{B,C} <: Transform
    Mins::B
    Maxes::C
end

"""
    RangeNorm( Z )

Acquires the minimum and maximum of each column in `Z` provided and returns a transform that performs the following operation (Z - min(X))/(max(X) - min(X)) on any future data. This has the important effect of scaling all values observed in the range of `Z` to be between 0 and 1 with respect to each column.
"""
function RangeNorm( Z )
    mins = minimum(Z, dims = 1)
    maxes = maximum(Z, dims = 1)
    RangeNorm( mins, maxes)
end

"""
    (T::RangeNorm)(Z; inverse = false)

Scales and shifts data in array `Z` column-wise according to learned min-maxes in RangeNorm object `T`.
"""
(T::RangeNorm)(Z; inverse = false) = (inverse) ? ( (Z .* ( T.Maxes .- T.Mins ) ) .+ T.Mins) : (Z .- T.Mins) ./ ( T.Maxes .- T.Mins )

struct BoxCox <: Transform
    innercall
end

"""
    BoxCox(lambda)

Returns a BoxCox transform operator/function. To be used in a pipeline.
"""
BoxCox(lambda) = return BoxCox(X; inverse = false) = begin
    Z = zeros(size(X))
    if inverse
        if lambda != 0.0
            Z =  ((X .* lambda).+ 1.0) .^ (1/lambda)
        else
            Z = exp.(X)
        end
    else
        if lambda != 0.0
            Z = ((X .^ lambda) .- 1.0) ./ lambda
        else
            Z = log.(X)
        end
    end
    return Z
end

"""
    Logit(Z; inverse = false)

Logit transforms (```ln( X / (1 - X) ))```) every element in `Z`. The inverse may also be applied.
*Warning: This can return Infs and NaNs if elements of Z are not suited to the transform*
"""
function Logit(Z; inverse = false)
    if inverse
        return (exp.(Z) ./ (1.0 .+ exp.(Z)))
    else
        return log.( (Z ./ (1.0 .- Z)))
    end
end
