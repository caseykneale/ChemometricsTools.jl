abstract type ClusterModel end

"""
    TotalClusterSS( Clustered::ClusterModel )

Returns a scalar of the total sum of squares for a ClusterModel object.
"""
function TotalClusterSS( Clustered::ClusterModel )
    GrandMean = StatsBase.mean( Clustered.X , dims = 1 )
    return sum((Clustered.X .- GrandMean).^ 2)
end

"""
    WithinClusterSS( Clustered::ClusterModel )

Returns a scalar of the within cluter sum of squares for a ClusterModel object.
"""
function WithinClusterSS( Clustered::ClusterModel )
    Clusters = unique( Clustered.Assignments )
    WithinSS = zeros( size(Clustered.X) )
    for cluster in Clusters
        ElementsInCluster = vec(Clustered.Assignments .== cluster)
        samples = Clustered.X[ ElementsInCluster, : ]
        clusterMean = StatsBase.mean( samples , dims = 1 )
        WithinSS[ElementsInCluster,:] = (samples .- clusterMean) .^ 2
    end
    return sum( WithinSS )
end

"""
    BetweenClusterSS( Clustered::ClusterModel )

Returns a scalar of the between cluster sum of squares for a ClusterModel object.
"""
function BetweenClusterSS( Clustered::ClusterModel )
    Clusters = unique( Clustered.Assignments )
    GrandMean = StatsBase.mean( Clustered.X , dims = 1 )
    BetweenSS = 0.0
    for cluster in Clusters
        ElementsInCluster = vec(Clustered.Assignments .== cluster)
        samples = Clustered.X[ ElementsInCluster, : ]
        clusterMean = StatsBase.mean( samples , dims = 1 )
        BetweenSS += size(samples)[1] * sum( ( clusterMean .- GrandMean ) .^ 2)
    end
    return BetweenSS
end

struct KMeansClustering <: ClusterModel
    X
    Centroids
    Assignments
end

"""
    KMeans( X, Clusters; tolerance = 1e-8, maxiters = 200 )

Returns a ClusterModel object after finding clusterings for data in `X` via MacQueens K-Means algorithm. `Clusters` is the K parameter, or the # of clusters.

MacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations. Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability. 1. University of California Press. pp. 281–297.
"""
function KMeans( X, Clusters; tolerance = 1e-8, maxiters = 200 )
    (Xrows, Xcols) = size( X )
    ResultVector = zeros( Xrows )
    #1) Initialize K-Means centroids: Randomly select C samples
    rndsmpl = collect( StatsBase.sample( collect(1 : Xrows), Clusters, replace = false ) )
    Centroids = X[ rndsmpl, :]
    newCentroid = zeros( Clusters, Xcols )
    #2) finding which many points are nearest to our centroid
    #3) Calculate the mean value of each dimension to make a new centroid and repeat.
    for iter in 1 : maxiters
        DistsToCentroids = SquareEuclideanDistance( X, Centroids )
        ResultVector = last.(Tuple.(argmin( DistsToCentroids, dims = 2 )))

        for cluster in 1:Clusters
            Assigned = (ResultVector .== cluster)[:,1]
            newCentroid[ cluster, : ] = StatsBase.mean( X[ Assigned, : ], dims = 1 )
        end
        ToleranceCheck = ( 1.0 / Clusters ) * sum( ( Centroids .- newCentroid ) .^ 2)
        if ToleranceCheck <= tolerance ; break; end
        Centroids = newCentroid
    end
    return KMeansClustering( X, Centroids, ResultVector )
end
