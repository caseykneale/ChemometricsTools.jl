function MakeIntervals( columns::Int, intervalsize::Int = 20 )
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
#Intervals = MakeInterval( 20, 3 );
function MakeIntervals( columns::Int, intervalsizes::Union{Array, Tuple} = [20, 50, 100] )
    Intervals = Dict()
    for interval in intervalsizes
        Intervals[interval] = MakeIntervals(columns,  interval)
    end
    return Intervals
end

# for Interval in MakeIntervals( 20, [3, 5, 10] )
#     println(first(Interval))
# end

#Weights regression outputs by their relative error
#Square error has mathematical gaurantees - so it's default..
function stackedweights(ErrVec; power = 2)
    SqErr = (1.0 ./ ErrVec) .^ power
    return SqErr / sum(SqErr)
end

struct RandomForest
    ensemble::Array{CART, 1}
end

function RandomForest(x, y, mode = :classification; gainfn = entropy, trees = 50,
                        maxdepth = 10,  minbranchsize = 5,
                        samples = 0.7, maxvars = nothing)
    (obs, vars) = size(x)
    bag = floor(obs * samples) |> Int
    if isa(maxvars, Nothing)
        maxvars = floor( (classification) ? sqrt(vars) : (vars / 3.0) ) |> Int
    end

    Forest = []
    for tree in 1:trees
        grabbag = unique( rand( 1:obs, bag ) )
        if mode == :classification
            push!(Forest, ClassificationTree(x[grabbag,:], y[grabbag,:]; gainfn = entropy,
                        maxdepth = maxdepth, minbranchsize = minbranchsize, varsmpl = maxvars))
        else
            push!(Forest, RegressionTree(x[grabbag,:], y[grabbag]; gainfn = entropy,
                        maxdepth = maxdepth, minbranchsize = minbranchsize, varsmpl = maxvars))
        end
    end
    return RandomForest(Forest)
end

function (RF::RandomForest)(X)
    (Obs, Vars) = size(X)
    Predictions = zeros(Obs, RF.ensemble[1].MaxClasses)
    Trees = length(RF.ensemble)
    #Make good use of a running mean for leaner memory consumption and minimual fn calls...
    for tree in 1 : Trees
        Predictions .+= RF.ensemble[tree](X)
    end
    return Predictions ./ Trees
end
