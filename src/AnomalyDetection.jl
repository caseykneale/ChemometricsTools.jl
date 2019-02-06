using Distributions

#Unwritten...
function KNN_OneClass(X)

end

struct PCA_Hotelling
    pca
    Lambda
end

#Untested
function Q(X, PCAModel::PCA; Significance = 0.05, Variance = 0.99)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( PCAModel ) )
    PCs = sum(CumVar .<= Variance)

    Q = X * (LinearAlgebra.Diagonal(ones(Factors)) - PCA.Loadings[:,1:PCs] * PCA.Loadings[:,1:PCs]') * X'

    L = cumsum(PCA.SingularValues[ 1 : 3 ])
    H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
    Guass = quantile( Normal(), 1.0 - Significance )
    Upper = L[1] * ( ( ( H0 * Gauss * sqrt( 2.0 * L[2] ) ) / L[1] ) + 1.0 + ( ( L[2] * H0 * ( H0 - 1.0 )/ L[1]^2) )^(1.0/H0)
    return Q
end

#Untested
function Hotelling(X, PCAModel::PCA; Significance = 0.05, Variance = 0.99)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( PCAModel ) )
    PCs = sum(CumVar .<= Variance)
    #Truncate loadings & scores
    Loadings = PCA.Loadings[:,1:PCs]
    Scores = PCA.Scores[:,1:PCs]
    #Hotelling Statistic: T 2 = (X^T) W Λ − 1 W ˆ T X
    #Λ ˆ = diag ( l 1 , l 2 ,.., l l )
    Lambda = sqrt.( LinearAlgebra.Diagonal( 1.0 ./ ( PCA.SingularValues[1:PCs] ) ) )
    Tsq = X * Loadings * Lambda * Loadings' * X
    if Obs < 100
        Scalar = ( Factors * (Obs - 1) ) / ( Obs - Factors )
        Threshold = Scalar * (1.0 - Distributions.fdistpdf( Fstat, Factors, Obs - Factors ))
    else
        Threshold = cquantile( Chisq( Factors ), 1.0 - Significance)
    end
    return Tsq
end

function (Z::PCA_Hotelling)(X, Significance )
    (Obs,Vars) = size(X)
    Tsq = X' * Z.Loadings * Z.Loadings' * X
    XP = X * Loadings
    Residual = Z - X
    if size(X)[1] > 100#Chi square

    else#F distribution
        #Use PCA not traditional F test
        Fstat = ( (NROW(X) + Obs - Vars - 1) * Tsq ) / ((NROW(X) + Obs - 2) * Vars)
        Pval = 1.0 - Distributions.fdistpdf( Fstat, Vars, NROW(X) + Obs - Vars - 1 )
    end
    TaSq = (PCs * ( Obs - 1 ) / ( Obs - 1 )) * ChiSq
end
