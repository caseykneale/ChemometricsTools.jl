module ChemometricsTools

    include("Transformations.jl")
    export PipelineInPlace, Pipeline, Center, Scale, StandardNormalVariate, RangeNorm,
        MultiplicativeScatterCorrection, PCA_NIPALS, PCA, ExplainedVariance

    include("Analysis.jl")
    export CanonicalCorrelationAnalysis, BlandAltman

    include("AnomalyDetection.jl")
    export KNN_OneClass, PCA_Hotelling, Q, Hotelling

    include("ClassificationModels.jl")
    export Threshold, MulticlassThreshold, HighestVote, KNN,
        LinearDiscriminantAnalysis, ExplainedVariance

    include("Clustering.jl")
    export TotalClusterSS, WithinClusterSS, BetweenClusterSS,
        KMeansClustering, KMeans

    include("DistanceMeasures.jl")
    export SquareEuclideanDistance, EuclideanDistance, ManhattanDistance

    include("Ensembles.jl")
    export MakeIntervals

    include("Preprocess.jl")
    export FirstDerivative, SecondDerivative, FractionalDerivative, SavitzkyGolay,
        DirectStandardization, OrthogonalSignalCorrection
        #,TransferByOrthogonalProjection

    include("RegressionModels.jl")
    export ME, MAE, MAPE, SSE, MSE, RMSE, SSTotal, SSReg, SSRes, RSquare,
        PearsonCorrelationCoefficient, ClassicLeastSquares, RidgeRegression,
        PrincipalComponentRegression, PartialLeastSquares, ExtremeLearningMachine,
        PredictFn

    include("Sampling.jl")
    export KennardStone

    include("Training.jl")
    export Shuffle, VenetianBlinds, Shuffle!, LeaveOneOut, KFoldsValidation

#ToDo: Add tests...

end # module
