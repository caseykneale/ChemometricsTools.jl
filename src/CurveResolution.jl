using CSV
using DataFrames
using Plots
using LinearAlgebra
using StatsBase

Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/triliq.csv");
Fraud = collect(convert(Array, Raw)[:,1:end]');



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
    #Monitor change in F norm...
    while (sum( ( Last .- W * H ) .^ 2)  > tolerancesq) && (iters < maxiters)
        Last = W * H
        H .*= ( W' * X ) ./ ( W' * W * H )
        W .*= ( X * H' ) ./ ( W * H * H')
        iters += 1
    end
    return (W, H)
end

#This needs some pretty serious cleaning, and was really tricky to write...
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

function FNNLS(A, b; LHS = false,
                maxiters = 520)
    if LHS
        ATA = A * A'
        ATb = A * b'
    else
        ATA = A' * A
        ATb = A' * b
    end
    tolerance = 1e-12*sum(abs.(ATA))*size(ATA)[1]
    P = zeros( size( ATA )[ 2 ] ) .|> Int
    R = collect( 1 : length( ATb ) ) .|> Int
    r = zeros( size( ATA )[ 2 ] )
    X = zeros( 1, size( ATA )[ 2 ] )
    Inds = collect( 1 : length( ATb ) ) .|> Int
    Rinds = collect( 1 : length( ATb ) ) .|> Int

    W = ATb
    inneriters = 0
    zeroint = 0 |> Int
    while any( R .> zeroint ) && any(W[Rinds] .> tolerance)
        Rinds = Inds[ R .> zeroint ]
        j = Rinds[argmax( W[Rinds] )]
        P[j] = j ; R[j] = zeroint
        Pinds = Inds[ P .> zeroint ] ; Rinds = Inds[ R .> zeroint ]
        #Update R & P
        r[Pinds] = Base.inv( ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
        r[Rinds] .= 0.0
        while any(r[Pinds] .<= tolerance) && (inneriters < maxiters)
            Select = (r .<= tolerance) .& (P .> zeroint)
            alpha = reduce( min, X[Select] ./ ( X[Select] .- r[Select] ) )
            X .+= alpha .* ( r .- X )
            Select = Inds[(abs.(X) .< tolerance) .& ( P .!= zeroint )]
            R[Select] .= Select
            P[Select] .= zeroint
            Pinds = Inds[ P .> 0 ] ; Rinds = Inds[ R .> zeroint ]
            r[Pinds] = Base.inv( ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
            r[Rinds] .= 0.0
            inneriters += 1
        end
        X = r
        W = ATb .- (ATA * X)
    end

    return X
end

# a = reshape( [73,111,52,87, 7,4, 46,72,27,80,89 , 71], 4,3)
# b = [96,7, 68,10]
# FNNLS(a, b)

a = randn(4,4);
b = randn(4);
x = FNNLS( a,  b)

#Torture test...
counterrs = 0
for i in 1:10000
    a = randn(4,4);
    b = randn(4);
    x = FNNLS( a,  b)
    if any(x .< -1e-2)
        counterrs += 1
    end
end
counterrs

function MCRALS(X, C, S = nothing; norm = (false, false),
                Factors = 1, maxiters = 20,
                nonnegative = (true, true) )
    @assert all( isa.( [ C , S ], Nothing ) ) == false
    err = zeros(maxiters)
    #X ./= sum(X, dims = 2)
    D = X#zeros(size(X))
    isC = isa(C, Nothing)
    isS = isa(S, Nothing)
    C = isC ? zeros(size(X)[1], Factors) : C[:, 1:Factors]
    S = isS ? zeros(Factors, size(X)[2]) : S[1:Factors, :]
    C ./= norm[1] ? sum(C, dims = 2) : 1.0
    S ./= norm[2] ? sum(S, dims = 1) : 1.0
    for iter in 1 : maxiters
        if !isS
            if nonnegative[2]
                for obs in 1 : size( X )[ 1 ]
                    C[obs,:] = FNNLS(S, X[obs,:]'; LHS = true)
                end
            else
                C = X * LinearAlgebra.pinv(S)
            end
            C ./= norm[1] ? sum(C, dims = 2) : 1.0
            isC = false
            D = C * S
        end
        if !isC
            if nonnegative[1]
                for var in 1:size(X)[2]
                    S[:,var] = FNNLS(C, vec(X[:,var]))
                end
            else
                S = LinearAlgebra.pinv(C) * X
            end
            S ./= norm[2] ? sum(S, dims = 1) : 1.0
            isS = false
            D = C * S
        end
        err[iter] = sum( ( X .- D ) .^ 2 ) / prod(size(X))
    end
    return ( C, S, err )
end

( C2, S2, vars ) = SIMPLISMA(Fraud; Factors = 5, exclude = nothing)

( C, S, err ) = MCRALS(Fraud', nothing, C2[:,[1,3,5]]'; Factors = 3)
err;
plot(err)


plot(S')

plot(C)

( W, H ) = NMF(Fraud; Factors = 3, maxiters = 300, tolerance = 1e-9)
plot(W)
#
plot(H')


#https://etd.ohiolink.edu/!etd.send_file?accession=ohiou1051480564&disposition=inline
function SIMPLISMA(X; Factors = 1, alpha = 0.05, exclude = nothing)
    (obs, vars) = size(X)
    PurestVar = ones(Factors) .|> Int
    Ortho = zeros(obs, Factors)
    SSE = StatsBase.sum(X .^ 2, dims = 1)
    e = (SSE .- StatsBase.sum(X, dims = 1).^2) / obs #RSE/SSE
    if !isa(exclude, Nothing)
        e[exclude] .= -Inf
    end
    PurestVar[1] = argmax(vec(e))
    Intensity = X[:, PurestVar[1]]
    Ortho[:,1] = Intensity ./ sqrt( Intensity' * Intensity )#2-norm
    for F in 2 : Factors
        proj = sum( (Ortho[:,1:(F-1)]' * X) .^ 2, dims = 1)
        p = vec(e .* (1.0 .- proj ./ SSE))
        p[PurestVar[1:F]] .= -Inf
        if !isa(exclude, Nothing)
            e[exclude] .= -Inf
        end
        PurestVar[F] = argmax( p )
        Intensity = X[:, PurestVar[F]]
        OrthTmp = Intensity .- sum(proj * (proj' * Intensity'))
        Ortho[:,F] = OrthTmp ./ sqrt.(OrthTmp' * OrthTmp)
    end

    C = X[:, PurestVar]
    S = LinearAlgebra.pinv(C) * X
    magnitude = vec(sum(S.^2, dims = 2))
    for F in 1:Factors
        S[F,:] = (magnitude[F] <= 1e-8) ? (S[F,:] .* 0.0) : (S[F,:] ./ sqrt(magnitude[F]))
    end
    return (C, S, PurestVar)
end

(C,S, vars) = SIMPLISMA(Fraud; Factors = 5, exclude = nothing)

plot(C[:,[1,3,5]])


plot(S')
