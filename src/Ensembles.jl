"""
    MakeIntervals( columns::Int, intervalsize::Int )

Returns an 1-Array of intervals from the range: 1 - `columns` of size `intervalsize`.
"""
function MakeIntervals( columns::Int, intervalsize::Int )
    ColSize = columns
    intlen = floor(ColSize / intervalsize) |> Int64
    Remainder = ColSize % intervalsize
    Intervals = [ (1 + ((i-1)*intervalsize)):(i*intervalsize) for i in 1:intlen ]
    if Remainder <= (intlen / 2)
        Intervals[end] = Intervals[end][1] : ColSize
    else
        push!(Intervals, last(Intervals[end]) : ColSize)
    end
    return Intervals
end

"""
    MakeIntervals( columns::Int, intervalsize::Union{Array, Tuple})

Creates an Dictionary whose key is the interval size and values are an array of intervals from the range: 1 - `columns` of size `intervalsize`.
"""
function MakeIntervals( columns::Int, intervalsizes::Union{Array, Tuple}  )
    Intervals = Dict()
    for interval in intervalsizes
        Intervals[interval] = MakeIntervals(columns,  interval)
    end
    return Intervals
end

"""
    stackedweights(ErrVec; power = 2)

Weights stacked interval errors by the reciprocal `power` specified. Used for SIPLS, SISPLS, etc.

Ni, W. , Brown, S. D. and Man, R. (2009), Stacked partial least squares regression analysis for spectral calibration and prediction. J. Chemometrics, 23: 505-517. doi:10.1002/cem.1246
"""
function stackedweights(ErrVec; power = 2)
    SqErr = (1.0 ./ ErrVec) .^ power
    return SqErr / sum(SqErr)
end

struct RandomForest
    ensemble::Array{CART, 1}
end

"""
    RandomForest(x, y, mode = :classification; gainfn = entropy, trees = 50, maxdepth = 10,  minbranchsize = 5, samples = 0.7, maxvars = nothing)

Returns a classification (`mode` = :classification) or a regression (`mode` = :regression) random forest model.
The `gainfn` can be entropy or gini for classification or ssd for regression.
If the number of `maximumvars` is not provided it will default to sqrt(variables) for classification or variables/3 for regression.

The returned object can be used for inference by calling new data on the object as a function.

Breiman, L. Machine Learning (2001) 45: 5. https://doi.org/10.1023/A:1010933404324
"""
function RandomForest(x, y, mode = :classification; gainfn = entropy, trees = 50,
                        maxdepth = 10,  minbranchsize = 5,
                        samples = 0.7, maxvars = nothing)
    (obs, vars) = size(x)
    bag = floor(obs * samples) |> Int
    if isa(maxvars, Nothing)
        maxvars = floor( (mode == :classification) ? sqrt(vars) : (vars / 3.0) ) |> Int
    end

    Forest = []
    for tree in 1:trees
        grabbag = unique( rand( 1:obs, bag ) )
        if mode == :classification
            push!(Forest, ClassificationTree(x[grabbag,:], y[grabbag,:]; gainfn = gainfn,
                        maxdepth = maxdepth, minbranchsize = minbranchsize, varsmpl = maxvars))
        else
            push!(Forest, RegressionTree(x[grabbag,:], y[grabbag]; gainfn = gainfn,
                        maxdepth = maxdepth, minbranchsize = minbranchsize, varsmpl = maxvars))
        end
    end
    return RandomForest(Forest)
end

"""
    (RF::RandomForest)(X)

Returns bagged prediction vector of random forest model.
"""
function (RF::RandomForest)(X)
    (Obs, Vars) = size(X)
    Predictions = zeros(Obs, RF.ensemble[1].MaxClasses)
    Trees = length(RF.ensemble)
    for tree in 1 : Trees
        Predictions .+= RF.ensemble[tree](X)
    end
    return Predictions ./ Trees
end
