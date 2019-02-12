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



# Fast Non-Negative Least Squares algorithm based on Bro, R., & de Jong, S. (1997) A fast
#non-negativity-constrained least squares algorithm. Journal of Chemometrics, 11, 393-401.
# Input: A∈R m×n, b∈R mxOutput:
# x∗≥0 such thatx∗= arg min‖Ax−b‖2.
# Initialization:P=∅,R={1,2,···,n},x=0,w=ATb−(ATA)x
#repeat
# 1.  Proceed if R=/=∅ ∧ [maxi∈R(wi)> tolerance]
# 2.  j= arg maxi∈R(wi)
# 3.  Include the index j in P and remove it from R
# 4.  sP= [(ATA)P]−1(ATb)P
#     4.1.  Proceed if min(sP)≤0
#     4.2.  α=−mini∈P[xi/(xi−si)]
#     4.3.  x:=x+α(s−x)
#     4.4.  Update R and P
#     4.5.  sP= [(ATA)P]−1(ATb)P
#     4.6.  sR=0
# 5.x=s
# 6.w=AT(b−Ax)

using LinearAlgebra

function FNNLS(A, b; tolerance = 1e-5, maxiters = 50)
    ATA = A' * A
    ATb = A' * b
    P = zeros( size( ATA )[ 2 ] )
    R = convert.(Float64, collect( 1 : length( ATb ) ))
    r = zeros( size( ATA )[ 2 ] )
    X = zeros( 1, size( ATA )[ 2 ] )
    Inds = collect( 1 : length( ATb ) ) .|> Int
    W = ATb - (ATA * X')
    iter = 0
    breakcond = true
    while breakcond
        j = argmax( W )
        P[j] = R[j] ; R[j] = 0.0
        Pinds = Inds[ P .>= 1 ] .|> Int ; Rinds = Inds[ R .>= 1 ] .|> Int
        #Update R & P
        r[Pinds] = Base.inv(ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
        r[Rinds] .= 0.0

        if length(Pinds) > 0
            while any((r[Pinds]) .<= tolerance) && (iter < maxiters)
                iter += 1
                Select = (r .<= tolerance) .& (P .> 0.0)
                alpha = -1.0 * reduce( min, X[Select] ./ ( X[Select] .- r[Select] ) )
                X .+= alpha .* ( r .- X )
                Select = (abs.(X) .<= tolerance) .& (P .== 0.0)
                R[Select] .= Inds[Select] .|> Int
                P[Select] .= 0
                Pinds = Inds[ P .>= 1 ] .|> Int ; Rinds = Inds[ R .>= 1 ] .|> Int
                r[Pinds] = Base.inv(ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
                r[Rinds] .= 0.0
            end
        else
            iter = maxiters + 1
        end
        X = r
        W = ATb .- (ATA * X)

        breakcond = any( R .> 0.0 ) && (iter <= maxiters)
        if sum( R .> 0 ) > 0
            breakcond = breakcond && any(W[R .> 0] .> tolerance)
        end
    end

    return X
end

a = abs.(rand(4,4));
b = abs.(rand(4));

x = FNNLS( a,  b)

a*x
b
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
