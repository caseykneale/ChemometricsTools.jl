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
    export Transform, PipelineInPlace, Pipeline, Center, Scale, CenterScale,
        RangeNorm, Logit, boxcox

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
        DirectStandardization, OrthogonalSignalCorrection, MultiplicativeScatterCorrection,
        StandardNormalVariate, Scale1Norm, Scale2Norm, ScaleInfNorm, boxcarScaleMinMax,
        offsetToZero
        #,TransferByOrthogonalProjection

    include("RegressionModels.jl")
    export ClassicLeastSquares, RidgeRegression, PrincipalComponentRegression,
        PartialLeastSquares, KernelRidgeRegression, LSSVM, ExtremeLearningMachine, PredictFn

    include("Trees.jl")
    export OneHotOdds, entropy, gini, ssd, StumpOrNode, ClassificationTree, RegressionTree, CART

    include("Ensembles.jl")
    export MakeInterval, MakeIntervals, stackedweights, RandomForest

    include("Sampling.jl")
    export VenetianBlinds, SplitByProportion, KennardStone

    include("Training.jl")
    export Shuffle, Shuffle!, LeaveOneOut, KFoldsValidation

    include("PSO.jl")
    export PSO, Particle, Bounds

    include("CurveResolution.jl")
    export BTEMobjective, BTEM, NMF, SIMPLISMA, MCRALS, FNNLS

    include("PlottingTools.jl")
    export QQ, BlandAltman, plotchem, rectangle, IntervalOverlay


#ToDo: Add hundreds of unit tests...

end # module
