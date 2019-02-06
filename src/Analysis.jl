using LinearAlgebra
using StatsBase

function MatrixInverseSqrt(X, threshold = 1e-6)
    eig = eigen(X)
    diagelems = 1.0 ./ sqrt.( max.( eig.values , 0.0 ) )
    diagelems[ diagelems .== Inf ] .= 0.0
    return eig.vectors * LinearAlgebra.Diagonal( diagelems ) * Base.inv( eig.vectors )
end

#Untested...
struct CanonicalCorrelationAnalysis
    U
    V
    r
end

function CanonicalCorrelationAnalysis(A, B)
    (Obs,Vars) = size(A);
    CAA = (1/Obs) .* A * A'
    CBB = (1/Obs) .* B * B'
    CAB = (1/Obs) .* A * B'
    maxrank = min( LinearAlgebra.rank( A ), LinearAlgebra.rank( B ) )
    CAAInvSqrt = MatrixInverseSqrt(CAA)
    CBBInvSqrt = MatrixInverseSqrt(CBB)
    singvaldecomp = LinearAlgebra.svd( CAAInvSqrt * CAB * CBBInvSqrt )
    Aprime = CAAInvSqrt * singvaldecomp.U[ :,1 : maxrank ]
    Bprime = CAAInvSqrt * singvaldecomp.Vt[ :,1 : maxrank ]
    return CanonicalCorrelationAnalysis(Aprime' * A, V = Bprime' * B, singvaldecomp.S[1 : maxrank] )
end

#Untested...
struct BlandAltman
    means::Array{Float64, 1}
    differences::Array{Float64, 1}
    UpperLimit::Float64
    Center::Float64
    LowerLimit::Float64
    Outliers::Array{Float64}
end

function BlandAltman(Y1, Y2)
    means = (Y1 .+ Y2) ./ 2.0
    diffs = Y2 .- Y1
    MeanofDiffs = StatsBase.mean( diffs )
    StdofDiffs = StatsBase.std( diffs )

    UpperLimit = MeanofDiffs + bounds * StdofDiffs
    Center = MeanofDiffs
    LowerLimit = MeanofDiffs - bounds * StdofDiffs
    #To:Do Add trend-line....
    Outliers = findall( (diffs .> MeanofDiffs + 1.96*StdofDiffs) )
    Outliers = vcat(Outliers, findall( diffs < MeanofDiffs - 1.96*StdofDiffs ) )
    return BlandAltman( means, diffs, UpperLimit, Center, LowerLimit, Outliers )
end
