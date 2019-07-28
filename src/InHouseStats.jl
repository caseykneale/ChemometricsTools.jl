#This file will have 0 dependencies...

"""
    rbinomial( p, size... )

Makes an N-dimensional array of size(s) `size` with a probability of being a 1 over a 0 of 1 `p`.
"""
rbinomial( p, size... ) = map( x -> ( x < p ) ? 1 : 0, rand( size... ) )

"""
    EmpiricalQuantiles(X, quantiles)

Finds the column-wise `quantiles` of 2-Array `X` and returns them in a 2-Array of size `quantiles` by `variables`.
*Note: This copies the array... Use a subset if memory is the concern. *
"""
function EmpiricalQuantiles(X, quantiles)
    @assert all((quantiles .>= 0.0) .& (quantiles .< 1.0))
    ( obs, vars ) = size( X )
    quantilevalues = zeros( length( quantiles ), vars )
    for v in 1 : vars
        Potentials = unique(X[:, v])
        if length(Potentials) > 2
            quantilevalues[:, v] = Statistics.quantile( X[:,v], quantiles )
        else
            if length(Potentials) == 2
                quantilevalues[1:2, v] = Potentials[1 : 2]
            else
                quantilevalues[:, v] .= Potentials[1]
            end
        end
    end
    return quantilevalues
end

mutable struct RunningMean
    mu::Float64
    p::Int
end

"""
    RunningMean(x)

Constructs a running mean object with an initial scalar value of `x`.
"""
RunningMean(x) = RunningMean( x, 1 )
"""
    Update!(RM::RunningMean, x)

Adds new observation(`x`) to a RunningMean object(`RM`) in place.
"""
function Update!(RM::RunningMean, x)
    RM.p += 1
    RM.mu += (x - RM.mu) / RM.p
end

"""
    Update(RM::RunningMean, x)

Adds new observation(`x`) to a RunningMean object(`RM`) and returns the new object.
"""
Update(RM::RunningMean, x) = RunningMean( RM.mu + (x - RM.mu) / ( RM.p + 1 ), RM.p + 1 )

"""
    Remove!(RM::RunningMean, x)

Removes an observation(`x`) from a RunningMean object(`RM`) and reculates the mean in place.
"""
function Remove!(RM::RunningMean, x)
    RM.mu = (RM.p * RM.mu - x) / (RM.p - 1)
    RM.p -= 1
end

"""
    Remove!(RM::RunningMean, x)

Removes an observation(`x`) from a RunningMean object(`RM`) and recuturns the new RunningMean object.
"""
Remove(RM::RunningMean, x) = RunningMean( (RM.p * RM.mu - x) / (RM.p - 1), RM.p - 1 )


mutable struct RunningVar
    m::RunningMean
    v::Float64
end

"""
    RunningVar(x)

Constructs a RunningVar object with an initial scalar value of `x`.
*Note: RunningVar objects implicitly calculate the running mean.*
"""
RunningVar(x) = RunningVar( RunningMean( x, 1 ), 0.0 )

"""
    Update!(RV::RunningVar, x)

Adds new observation(`x`) to a RunningVar object(`RV`) and updates it in place.
"""
function Update!(RV::RunningVar, x)
    OldMean = copy(RV.m.mu)
    Update!(RV.m, x)
    RV.v = ( (RV.v * (RV.m.p - 2)) + ( (x - OldMean) * ( x - RV.m.mu ) ) ) / (RV.m.p - 1.0)
end

"""
    Variance(rv::RunningVar)

Returns the current variance inside of a RunningVar object.
"""
Variance(rv::RunningVar) = rv.v

"""
    Mean(rv::RunningVar)

Returns the current mean inside of a RunningVar object.
"""
Mean(rv::RunningVar) = rv.m.mu

"""
    Mean(rv::RunningMean)

Returns the current mean inside of a RunningMean object.
"""
Mean(rm::RunningMean) = rm.mu

"""
    Skewness(X)

returns a measure of skewness for a population vector `X`.

Joanes, D. N., and C. A. Gill. 1998. “Comparing Measures of Sample Skewness and Kurtosis”. The Statistician 47(1): 183–189.
"""
Skewness(X) = (sum( (X .- Statistics.mean(X)) .^ 3) / length(X)) / (Statistics.var(X) ^ (1.5))

"""
    SampleSkewness(X)

returns a measure of skewness for vector `X` that is corrected for a sample of the population.

Joanes, D. N., and C. A. Gill. 1998. “Comparing Measures of Sample Skewness and Kurtosis”. The Statistician 47(1): 183–189.
"""
function SampleSkewness(X)
    N = length(X)
    @assert N > 2
    return ( sqrt( N * (N - 1) ) / (N - 2) ) * Skewness( X )
end

#This was written for an algorithm and didn't fit in anywhere so for now it's kept
#but it may not have use...
struct PermutedVectorPair{A,B,C}
    vec1::A
    vec2::B
    operation::C
    i::Int
    length::Int
end

"""
    PermutedVectorPair(vec1, vec2; op = +)

Returns an iterator which applies each element in vec2 to vec1 via the user selected operator(op)
"""
function PermutedVectorPair(vec1, vec2; op = +)
    return PermutedVectorPair(vec1, vec2, op, 1, length(vec2))
end

function Base.iterate(iterator::PermutedVectorPair, state = 1)
    if state > iterator.length
        return nothing
    else
        return ( broadcast(iterator.operation, iterator.vec1, iterator.vec2[state]) , state + 1)
    end
end

# for i in PermutedVectorPair([1,2,3],ones(10) )
#     println(i)
# end
