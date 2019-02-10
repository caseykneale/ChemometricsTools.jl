module ChemometricsTools
    include("ClassificationMetrics.jl")
    export LabelEncoding, HotToCold, ColdToHot, MulticlassStats,
        Threshold, MulticlassThreshold, HighestVote

    include("RegressionMetrics.jl")
    export ME, MAE, MAPE, SSE, MSE, RMSE, SSTotal, SSReg, SSRes, RSquare,
        PearsonCorrelationCoefficient, PercentRMSE

    include("DistanceMeasures.jl")
    export SquareEuclideanDistance, EuclideanDistance, ManhattanDistance

    include("Transformations.jl")
    export Transform, PipelineInPlace, Pipeline, Center, Scale, StandardNormalVariate,
        RangeNorm, Logit, MultiplicativeScatterCorrection

    include("Analysis.jl")
    export PCA_NIPALS, PCA, LDA, CanonicalCorrelationAnalysis, ExplainedVariance,
        BlandAltman, QQ

    include("AnomalyDetection.jl")
    export OneClassJKNN, Q, Hotelling, Leverage

    include("ClassificationModels.jl")
    export KNN, GaussianDiscriminant, LogisticRegression, MultinomialSoftmaxRegression

    include("Clustering.jl")
    export TotalClusterSS, WithinClusterSS, BetweenClusterSS,
        KMeansClustering, KMeans

    include("Preprocess.jl")
    export FirstDerivative, SecondDerivative, FractionalDerivative, SavitzkyGolay,
        DirectStandardization, OrthogonalSignalCorrection
        #,TransferByOrthogonalProjection

    include("RegressionModels.jl")
    export ClassicLeastSquares, RidgeRegression, PrincipalComponentRegression,
        PartialLeastSquares, ExtremeLearningMachine, PredictFn

    include("Ensembles.jl")
    export MakeIntervals

    include("Sampling.jl")
    export VenetianBlinds, SplitByProportion, KennardStone

    include("Training.jl")
    export Shuffle, Shuffle!, LeaveOneOut, KFoldsValidation

#ToDo: Add hundreds of unit tests...

end # module
