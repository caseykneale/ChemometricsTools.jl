"""
    OneClassJKNN( Normal, New; J::Int = 1, K::Int = 1, DistanceType = "euclidean" )

Creates a one class JK-NN classifier from `Normal` data and evaluates it on `New` data. This compares the inter sample
distance (`DistanceType`) between a `New` and `Normal` J nearest neighbors to the K nearest neighbors of those
J nearest neighbors in the `Normal` set. No cut off is provided, that should be done by the end-user. A typical
cut off value is 1.0 .
"""
#ToDo: Only find distances of K's that matter! This will speed things up and slim memory use.
function OneClassJKNN( Normal, New; J::Int = 1, K::Int = 1, DistanceType = "euclidean" )
    Obs = size( Normal )[ 1 ]
    ObsNew = size( New )[ 1 ]
    DistMat = zeros( Obs, ObsNew )
    JNNPool = zeros( ObsNew, J )
    IntraDataMeanDist = zeros(ObsNew)
    #Apply Distance Fn
    if DistanceType == "euclidean"
        DistMat = SquareEuclideanDistance( Normal, New )
    elseif DistanceType == "manhattan"
        DistMat = ManhattanDistance( Normal, New )
    end
    #Find J nearest neighbors between new samples and the old samples
    for obs in 1 : ObsNew
        JNNPool[ obs , : ] = sortperm( DistMat[ : , obs ] )[ 1 : J ]
        IntraDataMeanDist[obs] = sum( DistMat[ JNNPool[ obs , : ] .|> Int, obs] )
    end
    #Find K nearest neighbors in the original/normal data
    InterDataDists = zeros( Obs, Obs )
    if DistanceType == "euclidean"
        InterDataDists = SquareEuclideanDistance( Normal )
    elseif DistanceType == "manhattan"
        InterDataDists = ManhattanDistance( Normal )
    end
    #For each new sample, find it's J nearest neighbors K nearest neighbor distances
    MeanDists = zeros( ObsNew )
    for obs in 1 : ObsNew
        #Find row where J lives ( ObsNew, J )
        Smpls = JNNPool[ obs , : ] .|> Int
        for j in 1 : J
            KNNInds = sortperm( InterDataDists[  : , Smpls[ j ] ] )
            Screened = (KNNInds[1] == Smpls[j]) ? KNNInds[ 2 : ( K + 1 ) ] : KNNInds[ 1 : K ]
            MeanDists[ obs ] += sum( InterDataDists[ Smpls, Screened ] )
        end
    end

    MeanDists ./= J #* K #No need  to divide by K
    return (IntraDataMeanDist ./ MeanDists)
end


"""
    Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)

Computes the hotelling Tsq and upper control limit cut off of a `pca` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`.

A review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere
https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
"""
function Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum( CumVar .<= Variance )
    @assert PCs > 0
    #Truncate loadings & scores
    Scores = pca(X; Factors = PCs)
    #Hotelling Statistic: T 2 = (X^T) W Λ − 1 W ˆ T X
    #Λ ˆ = diag ( l 1 , l 2 ,.., l l )
    Lambda = sqrt.( LinearAlgebra.Diagonal( 1.0 ./ ( pca.Values[1:PCs] ) ) )
    #We only want to compare the Tsq between each element
    #but it is convenient to calculate everything at once...
    Tsq = diag( Scores * Lambda * Scores' )
    if Obs < 100#vetted this threshold, the calculations are correct.
        Scalar = ( PCs * ((Obs^2) - 1) ) / ( Obs * ( Obs - PCs ) )
        Threshold = Scalar * quantile(Distributions.FDist( PCs, Obs - PCs ), Quantile)
    else
        Threshold =  quantile( Chisq( PCs ),  Quantile )
    end
    return Tsq, Threshold
end

"""
    Leverage(pca::PCA)

Calculates the leverage of samples in a `pca` object.
"""
function Leverage(pca::PCA)
    return [ sum(diag(pca.Scores[r,:] * Base.inv(pca.Scores[r,:]' * pca.Scores[r,:]) * pca.Scores[r,:]')) for r in 1:size(pca.Scores)[1] ]
end

"""
    Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)

Computes the Q-statistic and upper control limit cut off of a `pca` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`.

A review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere
https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
"""
function Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum(CumVar .<= Variance)
    Q = diag(X * (LinearAlgebra.Diagonal(ones(Vars)) - pca.Loadings[1:PCs,:]' * pca.Loadings[1:PCs,:]) * X')
    L = [ sum(pca.Values[(PCs + 1) : end] .^ order) for order in [ 1, 2, 3 ] ]
    H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
    Gauss = quantile( Normal(),  Quantile )
    FirstTerm = ( Gauss * sqrt( 2.0 * L[2] * (H0^2.0) ) ) / L[1]
    SecondTerm =  (L[2] * H0 * ( 1.0 - H0 ) ) / (L[1] ^ 2.0)
    Upper = L[1] * ( FirstTerm + 1.0 + SecondTerm ) ^ 2.0
    return Q, Upper
end
