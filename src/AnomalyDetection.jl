using Distributions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Analysis.jl")

#Unwritten...
function KNN_OneClass(X)
    println("Coming soon...")
end

struct PCA_Hotelling
    pca
    Lambda
end


#Incomplete
#https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
function Q(X, pca::PCA; Significance = 0.05, Variance = 0.99)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum(CumVar .<= Variance)
    Q = X * (LinearAlgebra.Diagonal(ones(Vars)) - pca.Loadings[1:PCs,:]' * pca.Loadings[1:PCs,:]) * X'
     #Correct up to this point... The rest well... Gonna need to figure it out
     # L = cumsum(pca.Values[ 1 : 3 ])
     # H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
     # Gauss = quantile( Normal(), 1.0 - Significance )
     # Upper = L[1] * ( ( ( H0 * Gauss * sqrt( 2.0 * L[2] ) ) / L[1] ) + 1.0 + ( ( L[2] * H0 * ( H0 - 1.0 )/ L[1]^2) )^(1.0/H0))
    return Q#, Upper
end
Q(test, pca_obj; Variance = 1.0)

#Untested
#https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
function Hotelling(X, pca::PCA; Significance = 0.05, Variance = 1.0)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum( CumVar .<= Variance )
    #Truncate loadings & scores
    #Loadings = pca.Loadings[1:PCs,:]
    Scores = pca.Scores[:,1:PCs]
    #Hotelling Statistic: T 2 = (X^T) W Λ − 1 W ˆ T X
    #Λ ˆ = diag ( l 1 , l 2 ,.., l l )
    Lambda = sqrt.( LinearAlgebra.Diagonal( 1.0 ./ ( pca.Values[1:PCs] ) ) )
    #We only want to compare the Tsq between each element
    #but it is convenient to calculate everything at once...
    Tsq = diag( Scores * Lambda * Scores' )
    if Obs < 100
        Scalar = ( PCs * (Obs - 1) ) / ( Obs - PCs )
        Threshold = Scalar * quantile(Distributions.FDist( PCs, Obs - PCs ), Significance)
    else
        Threshold = quantile( Chisq( PCs ),  Significance)
    end
    return Tsq, Threshold
end

function Leverage(pca::PCA)
    return [ sum(diag(pca.Scores[r,:] * Base.inv(pca.Scores[r,:]' * pca.Scores[r,:]) * pca.Scores[r,:]')) for r in 1:size(pca.Scores)[1] ]
end

# using Plots
# test = vcat( randn(100,4), 7 .* randn(3,4) ) ;
#
# pca_obj = PCA(test; Factors = 2)
#
# (tsq, t) = Hotelling(test, pca_obj; Variance = 1.0)
#
# scatter( pca_obj.Scores[:,1], pca_obj.Scores[:,2])
# a = plot(tsq);
# Plots.abline!(a, 0, t, label = "Control Limit")
