#This file will have 0 dependencies...

#This copies the array... Use a subset if memory is the concern...
function EmpiricalQuantiles(X, quantiles)
    @assert all((quantiles .>= 0.0) .& (quantiles .< 1.0))
    ( obs, vars ) = size( X )
    quantilevalues = zeros( length( quantiles ), vars )
    for v in 1 : vars
        Potentials = sort( unique(X[:, v]) )

        for (j, potential) in enumerate(Potentials)
            lt = sum(X[:,v] .<= potential)
            ltscore = lt ./ obs
            for ( i, q ) in enumerate(quantiles)
                if (ltscore >= q) && ( quantilevalues[i, v] == 0.0)
                    quantilevalues[i, v] = potential
                end
            end
        end
    end
    return quantilevalues
end

# X = randn(3000,30);
# quantiles = (0.05, 0.5, 0.95)
# EmpiricalQuantiles(X, quantiles)

mutable struct RunningMean
    mu
    p
end
#Constructor for scalar
RunningMean(x) = RunningMean( x, 1 )

function Update!(RM::RunningMean, x)
    RM.p += 1
    RM.mu += (x - RM.mu) / RM.p
end

# using Statistics
# x = randn(100);
#
# Statistics.mean(x)
#
# z = RunningMean(x[1])
# for i in 2:100
#     Update!(z, x[i])
# end
# z
