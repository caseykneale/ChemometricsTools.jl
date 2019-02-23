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

struct CenterScale{B,C} <: Transform
    Mean::B
    StdDev::C
end

function CenterScale(Z)
    mu = StatsBase.mean(Z, dims = 1)
    stdev = StatsBase.std(Z, dims = 1)
    CenterScale( mu, stdev)
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

#I'll be the first to admit the methods below are not convenient...
#But it is the easiest way to include transforms into pipelines that have no learned parameters...
struct Logit <: Transform
     innercall
end
Logit(X) = return Logit(Z; inverse = false) = (inverse) ? (exp.(Z) ./ (1.0 .+ exp.(Z))) : log.( Z ./ (1.0 .- Z) )
