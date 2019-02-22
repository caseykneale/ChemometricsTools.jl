#Work in progress....

pwd()
cd("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/")
using Pkg
Pkg.activate(".")
using ChemometricsTools
using Plots
using CSV
using DataFrames
using Statistics

entropy(v) = -sum( map( x -> x * (x == 1.0 ? 0.0 : log( x , 2 )), v ) )

#find max gain, aka stump...
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

x2 = reshape([1,2, 1.2,1.4, 2.2, 1,2,3,4,5], 5,2)
yt = [1,1 ,2,2, 3];
tenc = LabelEncoding(yt)
hot = ColdToHot(yt, tenc);
hot
StumpOrNode(x2, hot)
tree(x2, hot; gainfn = entropy, maxdepth = 9, minbranchsize = 2 )



function OneHotOdds(Y)
    rsums = sum(Y, dims = 1);
    return Tuple(rsums ./ sum(rsums))
end

function tree(x, y; gainfn = entropy, maxdepth = 4, minbranchsize = 3)
    curdepth = 1 #Place holder for power of 2 depth of the binary tree
    cursky = 1 #Holds a 1 if branch can grow, 0 if it cannot
    curmap = [1 : size(y)[1]] #Holds indices available to the next split decision
    dt = []#Stores alllll of the decisions we make
    while (curdepth <= maxdepth) && (cursky >= 1 )
        nextmap = []
        nextsky = 0
        curdt = Dict()
        for sky in 1:cursky
            cmap = curmap[sky]#Get indices from last split for this partition
            if (curdepth == maxdepth) || (length(cmap) <= minbranchsize)#Truncate tree we are at our depth limit
                curdt[sky] = OneHotOdds( y[ cmap, : ] )#sky
            elseif length(cmap) > minbranchsize
                (bound, var) = StumpOrNode( x[cmap,:], y[cmap,:] ; gainfn = entropy )
                LHS = cmap[findall(x[cmap, var] .< bound)]
                RHS = cmap[findall(x[cmap, var] .>= bound)]
                if (length(LHS) >= 0) && (length(RHS) >= 0)
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
    return dt
end

function predict(x, dt, classes = 3)
    (Obs, Vars) = size(x) .|> Int
    output = zeros(Obs, classes)
    for obs in 1 : Obs
        Branch = 1
        LastBranch = 0

        Offset = 0
        LastOffset = 0
        for (i, d) in enumerate(dt)
            LastOffset = Offset
            if i > 1
                Offset = sum(map( x -> !isa(dt[i ][x], Pair), 1 : (Branch) ))
            end
            if haskey(d, Branch)
                if isa(d[Branch], Pair)
                    #println("off: ",Offset,"...",x[ obs, first(d[ Branch ]) ], "?", last(d[ Branch ]) )
                    if x[ obs, first(d[ Branch ]) ] < last(d[ Branch ])
                        LastBranch = Branch #- LastOffset
                        Branch = (Branch - Offset) + (Branch - Offset) - 1# - Offset
                    else
                        LastBranch = Branch #- LastOffset
                        Branch = (Branch - Offset) + (Branch - Offset)
                    end
                else
                    output[obs,:] .= d[Branch]# - Offset]
                    break
                end
            else
                println("Serious Error has Occurred...")
            end
        end
    end
    return output
end

Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/iris.data");
Lbls = convert(Array, Raw[1:(end-1),end]);
X = convert(Array,Raw[1:(end-1),1:(end-1)]);

Enc = LabelEncoding(Lbls);
Hot = ColdToHot(Lbls, Enc);

Shuffle!(X, Hot);
((TX,TY), (X,Y)) = SplitByProportion(X,Hot, 0.9)

@time dt = tree(TX,TY; gainfn = entropy, maxdepth = 9, minbranchsize = 5 )


@time q = predict(TX, dt, 3);
using StatsBase

sum(abs.(HighestVoteOneHot(q) .- TY))
MulticlassStats(HighestVoteOneHot(q), TY, Enc)

MulticlassStats(HighestVoteOneHot(predict(X, dt, 3)), Y, Enc)
sum(HighestVoteOneHot(predict(X, dt, 3)),dims = 1)
