#This file will have 0 dependencies...
rbinomial( p, size... ) = map( x -> ( x < p ) ? 1 : 0, rand( size... ) )

#This copies the array... Use a subset if memory is the concern...
function EmpiricalQuantiles(X, quantiles)
    @assert all((quantiles .>= 0.0) .& (quantiles .< 1.0))
    ( obs, vars ) = size( X )
    quantilevalues = zeros( length( quantiles ), vars )
    for v in 1 : vars
        Potentials = sort( unique(X[:, v]) )
        if length(Potentials) > 2
            for (j, potential) in enumerate(Potentials)
                lt = sum(X[:,v] .<= potential)
                ltscore = lt ./ obs
                for ( i, q ) in enumerate(quantiles)
                    if (ltscore >= q) && ( quantilevalues[i, v] == 0.0)
                        quantilevalues[i, v] = potential
                    end
                end
            end
        else
            if length(Potentials) == 2
                quantilevalues[1:2, v] = Potentials[1 : 2]
            else
                quantilevalues[1, v] = Potentials[1]
            end
        end
    end
    return quantilevalues
end

# X = randn(3000,30);
# quantiles = (0.05, 0.5, 0.95)
# EmpiricalQuantiles(X, quantiles)

mutable struct RunningMean
    mu::Float64
    p::Int
end
#Constructor for scalar
RunningMean(x) = RunningMean( x, 1 )

function Update!(RM::RunningMean, x)
    RM.p += 1
    RM.mu += (x - RM.mu) / RM.p
end

Update(RM::RunningMean, x) = RunningMean( RM.mu + (x - RM.mu) / ( RM.p + 1 ), RM.p + 1 )

#(n*un - xn)/(n − 1) =  μn−1
function Remove!(RM::RunningMean, x)
    RM.mu = (RM.p * RM.mu - x) / (RM.p - 1)
    RM.p -= 1
end

Remove(RM::RunningMean, x) = RunningMean( (RM.p * RM.mu - x) / (RM.p - 1), RM.p - 1 )

mutable struct RunningVar
    m::RunningMean
    v::Float64
end
#Constructor for scalar
RunningVar(x) = RunningVar( RunningMean( x, 1 ), 0.0 )
function Update!(RV::RunningVar, x)
    OldMean = copy(RV.m.mu)
    Update!(RV.m, x)
    RV.v = ( (RV.v * (RV.m.p - 2)) + ( (x - OldMean) * ( x - RV.m.mu ) ) ) / (RV.m.p - 1.0)
end

Variance(rv::RunningVar) = rv.v
Mean(rv::RunningVar) = rv.m.mu
Mean(rm::RunningMean) = rm.mu

using Statistics
x = randn(100);
Statistics.mean(x)
Statistics.var(x)
rv = RunningVar(x[1])

for i in 2:100
    Update!(rv, x[i])
end

rv

#using Statistics
#x = randn(100,10);

#Statistics.mean(x, dims = 1)
#Statistics.mean(x[1:49,:], dims = 1)

# z = RunningMean(x[1,:], 1);
# for i in 2:49
#     Update!(z, x[i,:])
# end
# z
# Statistics.mean(x[1:49]) - z.mu
