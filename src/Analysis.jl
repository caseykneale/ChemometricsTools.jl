using LinearAlgebra
using StatsBase

struct PCA
    Scores
    Loadings
    SingularValues
    algorithm::String
end

#NIPALS based PCA.
#Kind of advantageous is you don't want to outright compute all latent variables.
#Kind of slow, but a must for a chemometrics package...
function PCA_NIPALS(X; Factors = minimum(size(X)) - 1, tolerance = 1e-7, maxiters = 200)
    tolsq = tolerance * tolerance
    #Instantiate some variables up front for performance...
    Xsize = size(X)
    Tm = zeros( ( Xsize[1], Factors ) )
    Pm = zeros( ( Factors, Xsize[2] ) )
    t = zeros( ( 1, Xsize[1] ) )
    p = zeros( ( 1, Xsize[2] ) )
    #Set tolerance to floating point precision
    Residuals = copy(X)
    for factor in 1:Factors
        lastErr = sum(abs.(Residuals)); curErr = tolerance + 1;
        t = Residuals[:, 1]
        iterations = 0
        while (abs(curErr - lastErr) > tolsq) && (iterations < maxiters)
            p = Residuals' * t
            p = p ./ sqrt.( p' * p )
            t = Residuals * p
            #Track change in Frobenius norm
            lastErr = curErr
            curErr = sqrt(sum( ( Residuals - ( t * p' ) ) .^ 2))
            iterations += 1
        end
        Residuals -= t * p'
        Tm[:,factor] = t
        Pm[factor,:] = p
    end
    #Find singular values/eigenvalues
    EigVal = sqrt.( LinearAlgebra.diag( Tm' * Tm ) )
    #Scale loadings by singular values
    Tm = Tm * LinearAlgebra.Diagonal( 1.0 / EigVal )
    return PCA(Tm, Pm, EigVal, "NIPALS")
end

#SVD based PCA
function PCA(Z; Factors = minimum(size(Z)) - 1)
    svdres = LinearAlgebra.svd(Z)
    return PCA(svdres.U[:, 1:Factors], svdres.Vt[1:Factors, :], svdres.S[1:Factors], "SVD")
end

#Calling a PCA object on new data brings the new data into the PCA transforms basis...
(T::PCA)(Z::Array; Factors = length(T.SingularValues), inverse = false) = (inverse) ? Z * (Diagonal(T.SingularValues[1:Factors]) * T.Loadings[1:Factors,:]) : Z * (Diagonal( 1 ./ T.SingularValues[1:Factors]) * T.Loadings[1:Factors,:])'

ExplainedVariance(PCA::PCA) = ( PCA.SingularValues .^ 2 ) ./ sum( PCA.SingularValues .^ 2 )

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
    (Obs,Vars) = size(A);;
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
