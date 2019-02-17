using LinearAlgebra
using StatsBase
using Statistics

struct PCA
    Scores
    Loadings
    Values
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
(T::PCA)(Z::Array; Factors = length(T.Values), inverse = false) = (inverse) ? Z * (Diagonal(T.Values[1:Factors]) * T.Loadings[1:Factors,:]) : Z * (Diagonal( 1 ./ T.Values[1:Factors]) * T.Loadings[1:Factors,:])'

ExplainedVariance(PCA::PCA) = ( PCA.Values .^ 2 ) ./ sum( PCA.Values .^ 2 )

struct LDA
    Scores
    Loadings
    Values
end

function LDA(X, Y; Factors = 1)
    (Obs, ClassNumber) = size( Y )
    Variables = size( X )[ 2 ]
    #Instantiate some variables...
    ClassMeans = zeros( ClassNumber, Variables )
    ClassSize = zeros( ClassNumber )
    WithinCovariance = zeros( Variables, Variables )
    BetweenCovariance = zeros( Variables, Variables  )
    ClassCovariance = zeros( Variables, Variables  )
    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        ClassSize[class] = sum(Members)
        ClassMeans[class,:] = StatsBase.mean(X[Members,:], dims = 1)
    end
    GlobalMean = StatsBase.mean(ClassMeans, dims = 1)

    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        #calculate the between class covariance matrix
        Diff = (ClassMeans[class,:] .- GlobalMean)
        BetweenCovariance .+= (1.0 / (ClassSize[class] - 1.0)) .* ( Diff * Diff' )
        #calculate the within class covariance matrix
        MeanCentered = X[Members,:] .- ClassMeans[class, : ]'
        WithinCovariance .+= (1.0 / (ClassSize[class] - 1.0)) * ( MeanCentered' * MeanCentered  )
    end
    #Calculate the discriminant axis'
    eig = LinearAlgebra.eigen(Base.inv(WithinCovariance) * BetweenCovariance)
    if any( imag.( eig.values ) .> 1e-1)
        println("Warning: Some eigenvalues found to have complex contributions > 0.1")
    end
    #Maybe reccomend to the user to do pca first or centerscale or both?
    ReVals = real.(eig.values)
    Sorted = sortperm( ReVals, rev = true)
    Contributions = ReVals[Sorted] .>= 1e-9
    Loadings = real.(eig.vectors[:, Sorted[ Contributions] ] )
    #Project the X data into the LDA basis
    Scores = X * Loadings
    return LDA( Scores, Loadings, ReVals[ Sorted[ Contributions] ] )
end

function ( model::LDA )( Z; Factors = length(model.Values) )
     Projected = Z * model.Loadings[:,1:Factors]
end

ExplainedVariance(lda::LDA) = lda.Values ./ sum(lda.Values)


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
