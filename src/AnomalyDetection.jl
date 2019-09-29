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
