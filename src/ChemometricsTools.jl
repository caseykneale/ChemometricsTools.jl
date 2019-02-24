module ChemometricsTools
    using CSV
    using LinearAlgebra
    using Distributions#Could probably also get rid of this one...
    using Statistics
    using StatsBase
    using Plots
    using DSP #Ew I wanna get rid of this dependency... One function uses it...

    #A generic function that I use everywhere...
    forceMatrix(a) = (length(size(a)) == 1) ? reshape( a, length(a), 1 ) : a

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
        RangeNorm, Logit, BoxCox

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

    #Generic function for pulling data from within this package.
    #If enough datasets are provided then the data/dataloading could be a seperate package...
    datapath = joinpath(@__DIR__, "..", "data")
    ChemometricsToolsDatasets() = begin
        dircontents = readdir(datapath)
        return Dict( (1:length(dircontents)) .=> dircontents )
    end
    ChemometricsToolsDataset(filename::String) = CSV.read( Base.joinpath( datapath, filename ) )
    ChemometricsToolsDataset(file::Int) = CSV.read( Base.joinpath( datapath, readdir(datapath)[file] ) )
    export ChemometricsToolsDataset, ChemometricsToolsDatasets
    #ToDo: Add more unit tests to test/runtests.jl...

end # module
