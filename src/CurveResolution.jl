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
function FNNLS(A, b;
                tolerance = 1e-6 * prod( size( A ) ),
                maxiters = 150)
    eps = 1e-6
    ATA = A' * A
    ATb = A' * b
    P = zeros( size( ATA )[ 2 ] ) .|> Int
    R = collect( 1 : length( ATb ) ) .|> Int
    r = zeros( size( ATA )[ 2 ] )
    X = zeros( 1, size( ATA )[ 2 ] )
    Inds = collect( 1 : length( ATb ) ) .|> Int
    Rinds = collect( 1 : length( ATb ) ) .|> Int

    W = ATb .- (ATA * X')
    inneriters = 0
    iter = 0
    breakcond = any( R .> 0 ) && any(W[Rinds .> 0] .> tolerance)

    while breakcond
        Rinds = Inds[ R .> 0 ]
        j = Rinds[argmax( W[Rinds] )]
        P[j] = j ; R[j] = 0

        Pinds = Inds[ P .> 0 ] ; Rinds = Inds[ R .> 0 ]
        #Update R & P
        r[Pinds] = ATA[ Pinds, Pinds ] / ATb[ Pinds ]'
        r[Rinds] .= 0.0
        while any(r[Pinds] .<= tolerance) && (inneriters < maxiters)
            Select = (r .<= tolerance) .& (P .> 0)
            alpha = reduce( min, X[Select] ./ ( X[Select] .- r[Select] ) )
            X .+= alpha .* ( r .- X )
            Select = (abs.(X) .< tolerance) .& ( P .== 0 )
            R[Select] .= Inds[Select]
            P[Select] .= 0
            Pinds = Inds[ P .> 0 ] ; Rinds = Inds[ R .> 0 ]
            if length(Pinds) > 0
                r[Pinds] = ATA[ Pinds, Pinds ] / ATb[ Pinds ]'
            end
            r[Rinds] .= 0.0
            inneriters += 1
        end

        X = r
        W = ATb .- (ATA * X)

        breakcond = false
        if any( R .> 0 )
            breakcond = any(W[Rinds] .> tolerance)
        end
    end

    return X
end

#Torture test...
# for i in 1:10000
#     a = rand(4,4);
#     b = rand(4);
#     x = FNNLS( a,  b)
#     if any(x .< 0.0)
#         println("ahhh")
#     end
# end
using LinearAlgebra
#Trying to do this from memory.. X = S C
#So that. S = (XtX)-1 C && C = S (XtX)-1
function MCRALS(X, C, S = nothing;
                Factors = 1, maxiters = 100,
                nonnegative = false)
    @assert all( isa.( [ C , S ], Nothing ) ) == false
    err = zeros(maxiters)
    D = zeros(size(X))
    isC = isa(C, Nothing)
    isS = isa(S, Nothing)
    C = isC ? zeros(size(X)[1], Factors) : C[:, 1:Factors]
    S = isS ? zeros(Factors, size(X)[2]) : S[1:Factors, :]
    C = C ./ sum(C, dims = 1)
    S = S ./ sum(S, dims = 2)
    for iter in 1 : maxiters
        if !isS
            if nonnegative
                # for f in 1:Factors
                #     C[:,f] = FNNLS(D, S[f,:])
                # end
            else#C[o, F] = (S[F, v] * D[v, o])
                C = (S * LinearAlgebra.pinv(X) )'
            end

            isC = false
            D = (C * S)
        end
        if !isC
            if nonnegative
                # for f in 1:size(C)[2]
                #     S[f,:] = FNNLS(D, C[:,f])
                # end
            else#S[F, V] = C'[F, o] * D'[o, v]
                S = C' * LinearAlgebra.pinv(X)'
            end
            S = S ./ sum(S, dims = 2)
            isS = false
            D = (C * S)
        end
        err[iter] = sum( ( X .- D ) .^ 2 ) / prod(size(X))
    end
    return ( C, S, err )
end
size(Fraud)
(C,S, vars) = SIMPLISMA(Fraud; Factors = 4, exclude = nothing)
( C, S, err ) = MCRALS(Fraud[:,20:29]', nothing, C[:,[1,4]]'; Factors = 2)
plot(err)


plot(S')

plot(C)

using CSV
using DataFrames
Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/triliq.csv");
Fraud = collect(convert(Array, Raw)[:,1:end]');

using Plots
using LinearAlgebra
size(svd(Fraud).U)



(W, H) = NMF(Fraud; Factors = 3)
#plot(svd(Fraud).V[:,1:2])
plot(C)


( W, H ) = NMF(Fraud; Factors = 3, maxiters = 200, tolerance = 1e-6)
plot((H)')
#
plot(Fraud)

using StatsBase

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
vars

plot(C)


plot(S')
