

#Simplest NMF algorithm ever...
#Algorithms for non-negative matrix factorization. Daniel D. Lee. H. Sebastian Seung.
#NIPS'00 Proceedings of the 13th International Conference on Neural Information Processing Systems. 535-54
function NMF(X; Factors = 1, tolerance = 1e-7, maxiters = 200)
    (Obs, Vars) = size(X)
    W = abs.( randn( Obs , Factors ) )
    H = abs.( randn( Factors, Vars ) )

    Last = zeros(Obs,Vars)
    tolerancesq = tolerance ^ 2
    iters = 0
    while (sum( ( Last .- W * H ) .^ 2)  > tolerancesq) && (iters < maxiters)
        Last = W * H
        H .*= ( W' * X ) ./ ( W' * W * H )
        W .*= ( X * H' ) ./ ( W * H * H')
        iters += 1
    end
    return (W, H)
end

#
# using CSV
# using DataFrames
# Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/MIR_Fruit_purees.csv");
# Lbls = convert.(Bool, occursin.( "NON", String.(names( Raw )) )[2:end]);
# Dump = collect(convert(Array, Raw)[:,2:end]');
# Fraud = Dump[Lbls,:];
#
#
# (W,H) = NMF(Fraud; Factors = 3, maxiters = 200, tolerance = 1e-6)
#
# using Plots
#
# plot((H)')
#
# plot(Fraud')
