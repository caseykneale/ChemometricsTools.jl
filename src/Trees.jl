"""
    OneHotOdds(Y)

Calculates the odds of a one-hot formatted probability matrix. Returns a tuple.
"""
function OneHotOdds(Y)
    rsums = sum(Y, dims = 1);
    return Tuple(rsums ./ sum(rsums))
end

"""
    entropy(v)

Calculates the Shannon-Entropy of a probability vector `v`. Returns a scalar. A common gain function used in tree methods.
"""
entropy(v) = -sum( map( x -> x * (x == 1.0 ? 0.0 : log( x , 2 )), v ) )

"""
    gini(p)

Calculates the GINI coefficient of a probability vector `p`. Returns a scalar. A common gain function used in tree methods.
"""
gini(p) = 1.0 - sum( p .^ 2 )

"""
    ssd(p)

Calculates the sum squared deviations from a decision tree split. Accepts a vector of values, and the mean of that
 vector. Returns a scalar. A common gain function used in tree methods.
"""
ssd(split,mean) = sum( ( split .- mean ) .^ 2 )

#ssd(split) = sum( ( split .- mean( split ) ) .^ 2 )

#The following functions are internal methods.
#I have concerns about some of the performance here...
#Ideas:
#       sortedinds = sortperm(  x[ sortedinds , var ]  )
#       Look for one hot speed ups in X vars?
function StumpOrNode( x, y ; gainfn = entropy )
    maxgain = -Inf
    (decisionbound, decisionvar) = (0.0, 0)
    (Obs, Vars) = size( x )
    beforeinfo = gainfn( sum( y, dims = 1 ) ./ Obs )
    sortedinds = 1:Obs
    for var in 1 : Vars
        sortedinds = sortperm(  x[ : , var ]  )
        y = y[sortedinds,:]
        x = x[sortedinds,:]
        lhsprops = sum( y[1,:]', dims = 1 )
        rhsprops = sum( y[2:end,:], dims = 1 )
        for obs in 2 : ( Obs - 1 )
            lhsprops .+= y[obs,:]'
            rhsprops .-= y[obs,:]'
            LHS = gainfn( lhsprops ./ obs )
            RHS = gainfn( rhsprops ./ (Obs - obs) )
            curgain = beforeinfo - ( (obs/Obs) * LHS + ((Obs - obs)/Obs) * RHS)
            if curgain > maxgain
                maxgain         = curgain
                decisionbound   = (x[sortedinds[obs],var] + x[sortedinds[obs + 1], var]) / 2.0
                decisionvar     = var
            end
        end
    end
    return (decisionbound, decisionvar)
end

#I made this a lot more efficient at the cost of being less generic
#I'd apologize for that, but, I'd rather have this be an end-user function
#then a developer function...
function StumpOrNodeRegress( x, y ; gainfn = ssd )
    maxgain = -Inf
    (decisionbound, decisionvar) = (0.0, 0)
    (Obs, Vars) = size( x )
    totalmean = RunningMean( mean(y), Obs )
    beforeinfo = gainfn( y, totalmean.mu )
    sortedinds = 1:Obs
    for var in 1 : Vars
        sortedinds = sortperm(  x[ : , var ]  )
        y = y[sortedinds]
        x = x[sortedinds,:]
        lhmean = RunningMean(y[1])
        rhmean = Remove(totalmean, y[1])
        for obs in 2 : ( Obs - 1 )
            Update!( lhmean, y[obs] )
            Remove!( rhmean, y[obs] )
            LHS = gainfn( y[1 : obs]        , lhmean.mu )
            RHS = gainfn( y[(obs + 1) : end], rhmean.mu )
            curgain = beforeinfo - ( (obs/Obs) * LHS + ((Obs - obs)/Obs) * RHS)
            if curgain > maxgain
                maxgain         = curgain
                decisionbound   = (x[sortedinds[obs], var] + x[sortedinds[obs + 1], var]) / 2.0
                decisionvar     = var
            end
        end
    end
    return (decisionbound, decisionvar)
end


struct CART
    Tree::Array
    MaxClasses::Int64
    Classification::Bool
end

"""
    ClassificationTree(x, y; gainfn = entropy, maxdepth = 4, minbranchsize = 3)

Builds a CART object using either gini or entropy as a partioning method. Y must be a one hot encoded 2-Array.
Predictions can be formed by calling the following function from the CART object: (M::CART)(x).

*Note: this is a purely nonrecursive decision tree. The julia compiler doesn't like storing structs of nested things.
I wrote it the recursive way in the past and it was quite slow, I think this is true also
of interpretted languages like R/Python...So here it is, nonrecursive tree's!
"""
function ClassificationTree(x, y; gainfn = entropy, maxdepth = 4, minbranchsize = 3, varsmpl = 0)
    curdepth = 1 #Place holder for power of 2 depth of the binary tree
    cursky = 1 #Holds a 1 if branch can grow, 0 if it cannot
    (Obs, Classes) = size(y)
    curmap = [1 : Obs] #Holds indices available to the next split decision
    dt = []#Stores alllll of the decisions we make
    (Obs, Vars) = size(x)
    varsavail = 1 : Vars
    (bound, var) = (1.0,1)
    while (curdepth <= maxdepth) && (cursky >= 1 )
        nextmap = []
        nextsky = 0
        curdt = Dict()
        for sky in 1:cursky
            cmap = curmap[sky]#Get indices from last split for this partition
            if (curdepth == maxdepth) || (length(cmap) <= minbranchsize)#Truncate tree we are at our depth limit
                curdt[sky] = OneHotOdds( y[ cmap, : ] )
            elseif length(cmap) > minbranchsize
                if varsmpl > 0
                    varsavail = unique( rand(1:Vars, varsmpl))
                end
                (bound, var) = StumpOrNode( x[cmap,varsavail], y[cmap,:] ; gainfn = gainfn )
                if varsmpl > 0
                    var = varsavail[var]
                end
                LHS = cmap[findall(x[cmap, var] .< bound)]
                RHS = cmap[findall(x[cmap, var] .>= bound)]
                if (length(LHS) > 0) && (length(RHS) > 0)
                    push!(nextmap, LHS, RHS)
                    nextsky += 2
                    curdt[sky] = (var => bound)
                else
                    curdt[sky] = OneHotOdds( y[ cmap, : ] )
                end
            end
        end

        if length(curdt) > 0
            push!(dt, curdt) #update decision tree
            curmap = nextmap #update our mapped observations
            cursky = nextsky
            curdepth += 1
        else
            break
        end
    end
    return CART(dt, Classes, true)
end

"""
    RegressionTree(x, y; gainfn = ssd, maxdepth = 4, minbranchsize = 3)

Builds a CART object using ssd as a partioning method. Y must be a one column Array.
Predictions can be formed by calling the following function from the CART object: (M::CART)(x).

*Note: this is a purely nonrecursive decision tree. The julia compiler doesn't like storing structs of nested things.
I wrote it the recursive way in the past and it was quite slow, I think this is true also
of interpretted languages like R/Python...So here it is, nonrecursive tree's!
"""
function RegressionTree(x, y; gainfn = ssd, maxdepth = 4, minbranchsize = 3, varsmpl = 0)
    curdepth = 1 #Place holder for power of 2 depth of the binary tree
    cursky = 1 #Holds a 1 if branch can grow, 0 if it cannot
    Obs = size(y)[1]
    curmap = [1 : Obs] #Holds indices available to the next split decision
    dt = []#Stores alllll of the decisions we make
    (Obs, Vars) = size(x)
    varsavail = 1 : Vars
    (bound, var) = (1.0, 1)
    while (curdepth <= maxdepth) && (cursky >= 1 )
        nextmap = []
        nextsky = 0
        curdt = Dict()
        for sky in 1:cursky
            cmap = curmap[sky]#Get indices from last split for this partition
            if (curdepth == maxdepth) || (length(cmap) <= minbranchsize)#Truncate tree we are at our depth limit
                curdt[sky] = mean( y[ cmap ] )
            elseif length(cmap) > minbranchsize
                if varsmpl > 0
                    varsavail = unique( rand(1:Vars, varsmpl) )
                end
                (bound, var) = StumpOrNodeRegress( x[cmap,varsavail], y[cmap] ; gainfn = gainfn )
                if varsmpl > 0
                    var = varsavail[ var ]
                end
                LHS = cmap[findall(x[cmap, var] .< bound)]
                RHS = cmap[findall(x[cmap, var] .>= bound)]
                if (length(LHS) > 0) && (length(RHS) > 0)
                    push!(nextmap, LHS, RHS)
                    nextsky += 2
                    curdt[sky] = (var => bound)
                else
                    curdt[sky] = mean( y[ cmap ] )
                end
            end
        end

        if length(curdt) > 0
            push!(dt, curdt) #update decision tree
            curmap = nextmap #update our mapped observations
            cursky = nextsky
            curdepth += 1
        else
            break
        end
    end
    return CART(dt, 1, false)
end


"""
    (M::CART)(x)

This is a universal CART object predict function.
"""
function (M::CART)(x)
    (Obs, Vars) = size(x) .|> Int
    output = zeros(Obs, M.MaxClasses)
    for obs in 1 : Obs
        Branch = 1
        Offset = 0
        for (i, d) in enumerate(M.Tree)
            if i > 1
                Offset = sum(map( x -> !isa(d[x], Pair), 1 : (Branch) ))
            end
            if haskey(d, Branch)
                if isa(d[Branch], Pair)
                    if x[ obs, first(d[ Branch ]) ] < last(d[ Branch ])
                        Branch = (Branch - Offset) + (Branch - Offset) - 1
                    else
                        Branch = (Branch - Offset) + (Branch - Offset)
                    end
                else
                    output[obs,:] .= d[Branch]
                    break
                end
            end
        end
    end
    return output
end
