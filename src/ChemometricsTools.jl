module ChemometricsTools
    using DataFrames, LinearAlgebra, Statistics, StatsBase, SparseArrays, Plots
    using CSV: read
    using Distributions #Could probably also get rid of this one...

    #A generic function that I use everywhere to coerce a vecto dim 0 to a row vector...
    forceMatrix( a ) = ( length( size( a ) ) == 1 ) ? reshape( a, length(a), 1 ) : a
    forceMatrixT( a ) = ( length( size( a ) ) == 1 ) ? reshape( a, 1, length(a) ) : a
    export forceMatrix, forceMatrixT

    include("InHouseStats.jl") #Has Docs
    export EmpiricalQuantiles, Update!, Remove!, Update, Remove, RunningMean, RunningVar,
        Variance, Mean, rbinomial, Skewness, SampleSkewness

    include("ClassificationMetrics.jl") #Has Docs
    export LabelEncoding, IsColdEncoded, HotToCold, ColdToHot, MulticlassStats,
        Threshold, MulticlassThreshold, HighestVote, StatsFromTFPN, StatsDictToDataFrame

    include("RegressionMetrics.jl") #Has Docs
    export ME, MAE, MAPE, SSE, MSE, RMSE, SSTotal, SSReg, SSRes, RSquare,
        PearsonCorrelationCoefficient, PercentRMSE

    include("DistanceMeasures.jl") #Has Docs
    export SquareEuclideanDistance, EuclideanDistance, ManhattanDistance,
        GaussianKernel, CauchyKernel, LinearKernel, Kernel, NearestNeighbors,
        AdjacencyMatrix, InClassAdjacencyMatrix, OutOfClassAdjacencyMatrix

    include("Transformations.jl") #Has Docs: Box Cox Omitted for now...
    export Transform, PipelineInPlace, Pipeline, QuantileTrim, Center, Scale,
        CenterScale, RangeNorm, Logit, BoxCox

    include("Analysis.jl") #Has Docs
    export PCA_NIPALS, PCA, LDA, CanonicalCorrelationAnalysis, ExplainedVariance,
        findpeaks, RAFFT, AssessHealth

    include("AnomalyDetection.jl") #Has docs
    export OneClassJKNN, Q, Hotelling, Leverage

    include("ClassificationModels.jl") #Has docs
    export KNN, ProbabilisticNeuralNetwork, GaussianDiscriminant, LogisticRegression, MultinomialSoftmaxRegression,
        GaussianNaiveBayes, HighestVoteOneHot, ConfidenceEllipse, LinearPerceptronSGD, LinearPerceptronBatch,
        HLDA

    include("Clustering.jl") #Has Docs
    export TotalClusterSS, WithinClusterSS, BetweenClusterSS, KMeansClustering, KMeans

    include("Preprocess.jl") #Has Docs
    export FirstDerivative, SecondDerivative, FractionalDerivative, SavitzkyGolay,
        DirectStandardization, OrthogonalSignalCorrection, MultiplicativeScatterCorrection,
        StandardNormalVariate, Scale1Norm, Scale2Norm, ScaleInfNorm, ScaleMinMax,
        offsetToZero, boxcar, ALSSmoother, PerfectSmoother, CORAL, TransferByOrthogonalProjection

    include("RegressionModels.jl") # Has Docs
    export ClassicLeastSquares, OrdinaryLeastSquares, RidgeRegression, PrincipalComponentRegression,
        PartialLeastSquares, KernelRidgeRegression, LSSVM, ExtremeLearningMachine, PredictFn, sigmoid

    include("Trees.jl") #Has Docs: Omitted StumpOrNode & StumpOrNodeRegress
    export OneHotOdds, entropy, gini, ssd, ClassificationTree, RegressionTree, CART

    include("Ensembles.jl") #Has Docs
    export MakeInterval, MakeIntervals, stackedweights, RandomForest

    include("Sampling.jl") #Has Docs
    export VenetianBlinds, SplitByProportion, KennardStone

    include("Training.jl") #Has Docs
    export Shuffle, Shuffle!, LeaveOneOut, KFoldsValidation

    include("PSO.jl") #Has docs
    export PSO, Particle, Bounds

    include("CurveResolution.jl") #Has Docs
    export BTEMobjective, BTEM, NMF, SIMPLISMA, MCRALS, FNNLS

    include("PlottingTools.jl") #Has Docs
    export QQ, BlandAltman, plotchem, rectangle, IntervalOverlay

    include("TimeSeries.jl") #Has Docs: Omitted EchoStateNetwork Fns
    export RollingWindow, EchoStateNetwork, TuneRidge, PredictFn, EWMA, Variance, Limits

    include("MultiWay.jl") #Has Docss
    export MultiCenter, MultiScale, MultiNorm, MultiPCA

    include("KernelDensityGenerator.jl") #Has Docs
    export Universe, GaussianBand, LorentzianBand

    include("SimpleGAs.jl") #No Docs yet :(
    export BinaryLifeform, Lifeform, SinglePointCrossOver, Mutate

    #Generic function for pulling data from within this package.
    #If enough datasets are provided then the data/dataloading could be a seperate package...
    #This will remain hard-coded until I have atleast 2 datasets that require permissions...
    TecatorStatement = "Statement of permission from Tecator (the original data source).These data are recorded on a Tecator" *
    "\n Infratec Food and Feed Analyzer working in the wavelength range 850 - 1050 nm by the Near Infrared" *
    "\n Transmission (NIT) principle. Each sample contains finely chopped pure meat with different moisture, fat" *
    "\n and protein contents.If results from these data are used in a publication we want you to mention the" *
    "\n instrument and company name (Tecator) in the publication. In addition, please send a preprint of your " *
    "\n article to Karin Thente, Tecator AB, Box 70, S-263 21 Hoganas, Sweden. The data are available in the " *
    "\n public domain with no responsability from the original data source. The data can be redistributed as long " *
    "\n as this permission note is attached. For more information about the instrument - call Perstorp Analytical's" *
    "\n representative in your area."

    datapath = joinpath(@__DIR__, "..", "data")
    ChemometricsToolsDatasets() = begin
        dircontents = readdir(datapath)
        dircontents = [ f for f in dircontents if f != "Readme.md" ]
        return Dict( (1:length(dircontents)) .=> dircontents )
    end
    function ChemometricsToolsDataset(filename::String)
        if filename == "tecator.csv"
            println(TecatorStatement)
        end
        if filename != "Readme.md"
            read( Base.joinpath( datapath, filename ) )
        else
            println("Don't load the markdown Readme as a csv... You're better than this.")
        end
    end
    function ChemometricsToolsDataset(file::Int)
        if readdir(datapath)[file] == "tecator.csv"
            println(TecatorStatement)
        end
        read( Base.joinpath( datapath, readdir(datapath)[file] ) )
    end
    export ChemometricsToolsDataset, ChemometricsToolsDatasets

end # module
