module ChemometricsTools
    include("ClassificationMetrics.jl")
    export LabelEncoding, IsColdEncoded, HotToCold, ColdToHot, MulticlassStats,
        Threshold, MulticlassThreshold, HighestVote

    include("RegressionMetrics.jl")
    export ME, MAE, MAPE, SSE, MSE, RMSE, SSTotal, SSReg, SSRes, RSquare,
        PearsonCorrelationCoefficient, PercentRMSE

    include("DistanceMeasures.jl")
    export SquareEuclideanDistance, EuclideanDistance, ManhattanDistance,
        GaussianKernel, LinearKernel, Kernel

    include("Transformations.jl")
    export Transform, PipelineInPlace, Pipeline, Center, Scale, StandardNormalVariate,
        RangeNorm, Logit

    include("Analysis.jl")
    export PCA_NIPALS, PCA, LDA, CanonicalCorrelationAnalysis, ExplainedVariance,
        findpeaks

    include("AnomalyDetection.jl")
    export OneClassJKNN, Q, Hotelling, Leverage

    include("ClassificationModels.jl")
    export KNN, GaussianDiscriminant, LogisticRegression, MultinomialSoftmaxRegression,
        GaussianNaiveBayes, HighestVoteOneHot

    include("Clustering.jl")
    export TotalClusterSS, WithinClusterSS, BetweenClusterSS,
        KMeansClustering, KMeans

    include("Preprocess.jl")
    export FirstDerivative, SecondDerivative, FractionalDerivative, SavitzkyGolay,
        DirectStandardization, OrthogonalSignalCorrection, MultiplicativeScatterCorrection
        #,TransferByOrthogonalProjection

    include("RegressionModels.jl")
    export ClassicLeastSquares, RidgeRegression, PrincipalComponentRegression,
        PartialLeastSquares, KernelRidgeRegression, LSSVM, ExtremeLearningMachine, PredictFn

    include("Ensembles.jl")
    export MakeInterval, MakeIntervals, stackedweights

    include("Sampling.jl")
    export VenetianBlinds, SplitByProportion, KennardStone

    include("Training.jl")
    export Shuffle, Shuffle!, LeaveOneOut, KFoldsValidation

    include("CurveResolution.jl")
    export NMF, SIMPLISMA, MCRALS, FNNLS

    include("PlottingTools.jl")
    export QQ, BlandAltman, plotchem, rectangle, IntervalOverlay

    include("Trees.jl")
    export OneHotOdds, entropy, gini,sse, StumpOrNode, ClassificationTree, RegressionTree, CART

#ToDo: Add hundreds of unit tests...

end # module
