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
        #pipe[i] = fn(X)
        pipe[i] = isa(fn, Function) ? fn : fn(X)
        X .= pipe[i]( X )
    end
    return pipeline(Tuple(pipe), true)
end

function Pipeline( X, FnStack...)
    pipe = Array{Any,1}(undef, length(FnStack))
    for (i, fn) in enumerate( FnStack )
        pipe[i] = isa(fn, Function) ? fn : fn(X)
        X = pipe[i]( X )
    end
    return pipeline(Tuple(pipe), false)
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
# using StatsBase
# FauxData2 = [1,1,2,3,4,5,6,7] ./ 10.0;
# Pipe1 = Pipeline(FauxData2,  Logit);
# RMSE( FauxData2, Pipe1(Pipe1(FauxData2); inverse = true) ) < 1e-14
