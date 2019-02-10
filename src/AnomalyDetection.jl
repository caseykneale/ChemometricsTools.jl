using Distributions


#Ideally I'd only access each column once in the K loop... Tricky...
#Untested. Not super optimal, but should work fine, for small data...
#Could store inter distance matrix to a struct...
#Untested...
function OneClassJKNN( Normal, New; J = 1, K = 1, threshold = 1.0 )
    Obs = size( Normal )[ 1 ]
    ObsNew = size( New )[ 1 ]
    DistMat = zeros( Obs, ObsNew )
    NNPool = zeros( ObsNew, J )
    IntraDataMeanDist = zeros(ObsNew)
    #Apply Distance Fn
    if model.DistanceType == "euclidean"
        DistMat = SquareEuclideanDistance( Obs, ObsNew )
    elseif model.DistanceType == "manhattan"
        DistMat = ManhattanDistance( Obs, ObsNew )
    end
    #Find nearest neighbors between new samples and the old samples
    for obs in 1 : ObsNew
        NNPool[ obs , : ] = sortperm( DistMat[ : , obs ] )[ 1 : J ]
        IntraDataMeanDist[obs] = sum( DistMat[ NNPool[ obs , : ], obs] )
    end
    #What samples do we need from the original dataset?
    OriginalSamples = unique( NNPool )
    #Make a dictionary to lookup indices later...
    Lookup = Dict( 1:length(OriginalSamples) .=> OriginalSamples  )
    OrigNNPool = zeros( Obs, Obs )
    InterDataMeanDist = zeros( Obs[ OriginalSamples ,:])
    if model.DistanceType == "euclidean"
        InterDataMeanDist = SquareEuclideanDistance( Obs )
    elseif model.DistanceType == "manhattan"
        InterDataMeanDist = ManhattanDistance( Obs )
    end

    for obs in 1 : ObsNew
        Smpls = NNPool[ obs , : ]
        for j in 1 : J
            InterDataMeanDist[ obs ] += sum( sort( InterDataMeanDist[ Smpls[j] , : ] )[ 1 : K ] )
        end
    end

    InterDataMeanDist[ obs ] ./= J #* K #No need  to divide by K
    return (IntraDataMeanDist ./ InterDataMeanDist) .< Threshold
end

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


#https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
function Q(X, pca::PCA; Significance = 0.05, Variance = 0.95)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum(CumVar .<= Variance)
    Q = diag(X * (LinearAlgebra.Diagonal(ones(Vars)) - pca.Loadings[1:PCs,:]' * pca.Loadings[1:PCs,:]) * X')
    #Correct up to this point... The rest well... Gonna need to figure it out
    L = [ sum(pca.Values[(PCs + 1) : end] .^ order) for order in [ 1, 2, 3 ] ]
    H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
    Gauss = quantile( Normal(), Significance )
    FirstTerm = ( Gauss * sqrt( 2.0 * L[2] * H0^2 ) ) / L[1]
    SecondTerm =  (L[2] * H0 * ( 1.0 - H0 ) ) / L[1]^2

    Upper = L[1] * ( FirstTerm + 1.0 + SecondTerm ) ^ (2)
    return Q, Upper
end
