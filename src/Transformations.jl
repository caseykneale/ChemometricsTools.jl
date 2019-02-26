abstract type Transform end

struct pipeline
    transforms
    inplace::Bool
end

#Naive Constructor...
Pipeline(Transforms) = pipeline(Transforms, false)

function PipelineInPlace( X, FnStack...)
    pipe = Array{Any,1}(undef, length(FnStack))
    for (i, fn) in enumerate( FnStack )
        pipe[i] = isa(fn, Function) ? fn : fn(X)
        X .= pipe[i]( X )
    end
    return pipeline(Tuple(pipe), true)
end

function Pipeline( X, FnStack... )
    pipe = Array{Any,1}( undef, length( FnStack ))
    #dumbyarray = randn(10, size(X)[2])
    for (i, fn) in enumerate( FnStack )
        pipe[ i ] = isa( fn, Function ) ? fn : fn( X )
        #Someone forgot the Transform tag...
        # if isdefined(pipe[i](dumbyarray), :invertible)
        #     pipe[ i ] = pipe[ i ]( X )
        # end
        X = pipe[ i ]( X )
    end
    return pipeline(Tuple( pipe ), false)
end

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


#Pipeline(randn(3,5), X -> QuantileTrim(X; quantiles = (0.2,0.8)), RangeNorm)
#Transforms with hyper parameters can be added via an anonymous function...
struct QuantileTrim <: Transform
    Quantiles::Array
    invertible::Bool
end

function QuantileTrim(Z; quantiles::Tuple{Float64,Float64} = (0.05, 0.95) )
    @assert length(quantiles) == 2
    return QuantileTrim( EmpiricalQuantiles(Z, quantiles), false )
end

function (T::QuantileTrim)(X, inverse = false)
    if inverse == false
        for c in 1:size(X)[2]
            lt = X[ : , c ] .< T.Quantiles[ 1, c ]
            gt = X[ : , c ] .> T.Quantiles[ 2, c ]
            X[ lt, c ] .= T.Quantiles[ 1, c ]
            X[ gt, c ] .= T.Quantiles[ 2, c ]
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

Center(Z) = Center( StatsBase.mean(Z, dims = 1), true )

#Call with new data transforms the new data, or inverts it
(T::Center)(Z; inverse = false) = (inverse) ? (Z .+ T.Mean) : (Z .- T.Mean)

struct Scale{B} <: Transform
    StdDev::B
    invertible::Bool
end
Scale(Z) = Scale( StatsBase.std(Z, dims = 1),  true )
(T::Scale)(Z; inverse = false) = (inverse) ? (Z .* T.StdDev) : (Z ./ T.StdDev)

struct CenterScale{B,C} <: Transform
    Mean::B
    StdDev::C
    invertible::Bool
end

function CenterScale(Z)
    mu = StatsBase.mean(Z, dims = 1)
    stdev = StatsBase.std(Z, dims = 1)
    CenterScale( mu, stdev, true)
end
#Call with new data transforms the new data, or inverts it
(T::CenterScale)(Z; inverse = false) = (inverse) ? ((Z .* T.StdDev) .+ T.Mean) : ((Z .- T.Mean) ./ T.StdDev)

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

struct BoxCox <: Transform
    innercall
end
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

function Logit(Z; inverse = false)
    if inverse
        return (exp.(Z) ./ (1.0 .+ exp.(Z)))
    else
        return log.( (Z ./ (1.0 .- Z)))
    end
end
