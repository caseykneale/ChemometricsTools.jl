var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Documentation",
    "title": "Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "#ChemometricsTools.BlandAltman-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.BlandAltman",
    "category": "method",
    "text": "BlandAltman(Y1, Y2; Confidence = 1.96)\n\nReturns a Plot object of a Bland-Altman plot between vectors Y1 and Y2 with a confidence limit of Confidence.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Bounds-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Bounds",
    "category": "method",
    "text": "Bounds(dims)\n\nConstructor for a Bounds object. Returns a bounds object with a lower bound of [lower...] and upper bound[upper...] with length of dims.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Bounds-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Bounds",
    "category": "method",
    "text": "Bounds(dims)\n\nDefault constructor for a Bounds object. Returns a bounds object with a lower bound of [0...] and upper bound[1...] with length of dims.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.CORAL-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.CORAL",
    "category": "method",
    "text": "CORAL(X1, X2; lambda = 1.0)\n\nPerforms CORAL to facilitate covariance based transfer from X1 to X2 with regularization parameter lambda. Returns a CORAL object.\n\nCorrelation Alignment for Unsupervised Domain Adaptation. Baochen Sun, Jiashi Feng, Kate Saenko. https://arxiv.org/abs/1612.01939\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.CORAL-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.CORAL",
    "category": "method",
    "text": "(C::CORAL)(Z)\n\nApplies a the transform from a learned CORAL object to new data Z.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ClassicLeastSquares-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ClassicLeastSquares",
    "category": "method",
    "text": "ClassicLeastSquares( X, Y; Bias = false )\n\nMakes a ClassicLeastSquares regression model of the form Y = AX with or without a Bias term. Returns a CLS object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ClassicLeastSquares-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ClassicLeastSquares",
    "category": "method",
    "text": "(M::ClassicLeastSquares)(X)\n\nMakes an inference from X using a ClassicLeastSquares object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianDiscriminant-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianDiscriminant",
    "category": "method",
    "text": "GaussianDiscriminant(M, X, Y; Factors = nothing)\n\nReturns a GaussianDiscriminant classification model on basis object M (PCA, LDA) and one hot encoded Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianDiscriminant-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianDiscriminant",
    "category": "method",
    "text": "( model::GaussianDiscriminant )( Z; Factors = size(model.ProjectedClassMeans)[2] )\n\nReturns a 1 hot encoded inference from Z using a GaussianDiscriminant object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianNaiveBayes-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianNaiveBayes",
    "category": "method",
    "text": "GaussianNaiveBayes(X,Y)\n\nReturns a GaussianNaiveBayes classification model object from X and one hot encoded Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianNaiveBayes-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianNaiveBayes",
    "category": "method",
    "text": "(gnb::GaussianNaiveBayes)(X)\n\nReturns a 1 hot encoded inference from X using a GaussianNaiveBayes object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KFoldsValidation-Tuple{Int64,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KFoldsValidation",
    "category": "method",
    "text": "KFoldsValidation(K::Int, x, y)\n\nReturns a KFoldsValidation iterator with K folds. Because it\'s an iterator it can be used in for loops, see the tutorials for pragmatic examples. The iterator returns a 2-Tuple of 2-Tuples which have the  following form: ((TrainX,TrainY),(ValidateX,ValidateY).\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KNN",
    "page": "Documentation",
    "title": "ChemometricsTools.KNN",
    "category": "type",
    "text": "KNN( X, Y; DistanceType::String )\n\nDistanceType can be \"euclidean\", \"manhattan\". Y Must be one hot encoded.\n\nReturns a KNN classification model.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KNN-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KNN",
    "category": "method",
    "text": "( model::KNN )( Z; K = 1 )\n\nReturns a 1 hot encoded inference from X with K Nearest Neighbors, using a KNN object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Kernel-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Kernel",
    "category": "method",
    "text": "(K::Kernel)(X)\n\nThis is a convenience function to allow for one-line construction of kernels from a Kernel object K and new data X.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LDA-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LDA",
    "category": "method",
    "text": "LDA(X, Y; Factors = 1)\n\nCompute\'s a LinearDiscriminantAnalysis transform from x with a user specified number of latent variables(Factors). Returns an LDA object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LDA-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LDA",
    "category": "method",
    "text": "( model::LDA )( Z; Factors = length(model.Values) )\n\nCalling a LDA object on new data brings the new data Z into the LDA basis.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LSSVM-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LSSVM",
    "category": "method",
    "text": "LSSVM( X, Y, Penalty; KernelParameter = 0.0, KernelType = \"linear\" )\n\nMakes a LSSVM model of the form Y = AK with a bias term using a user specified Kernel(\"Linear\", or \"Guassian\") and has an L2 Penalty. Returns a LSSVM Wrapper for a CLS object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LSSVM-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LSSVM",
    "category": "method",
    "text": "(M::LSSVM)(X)\n\nMakes an inference from X using a LSSVM object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LogisticRegression-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LogisticRegression",
    "category": "method",
    "text": "( model::LogisticRegression )( X )\n\nReturns a 1 hot encoded inference from X using a LogisticRegression object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MultiplicativeScatterCorrection-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.MultiplicativeScatterCorrection",
    "category": "method",
    "text": "(T::MultiplicativeScatterCorrection)(Z)\n\nApplies MultiplicativeScatterCorrection from a stored object T to Array Z.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.OrthogonalSignalCorrection-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.OrthogonalSignalCorrection",
    "category": "method",
    "text": "OrthogonalSignalCorrection(X, Y; Factors = 1)\n\nPerforms Thomas Fearn\'s Orthogonal Signal Correction to an endogenous X and exogenous Y. The number of Factors are the number of orthogonal components to be removed from X. This function returns an OSC object.\n\nTom Fearn. On orthogonal signal correction. Chemometrics and Intelligent Laboratory Systems. Volume 50, Issue 1, 2000, Pages 47-52.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.OrthogonalSignalCorrection-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.OrthogonalSignalCorrection",
    "category": "method",
    "text": "(OSC::OrthogonalSignalCorrection)(Z; Factors = 2)\n\nApplies a the transform from a learned orthogonal signal correction object OSC to new data Z.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PCA-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PCA",
    "category": "method",
    "text": "PCA(X; Factors = minimum(size(X)) - 1)\n\nCompute\'s a PCA from x using LinearAlgebra\'s SVD algorithm with a user specified number of latent variables(Factors). Returns a PCA object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PCA-Tuple{Array}",
    "page": "Documentation",
    "title": "ChemometricsTools.PCA",
    "category": "method",
    "text": "(T::PCA)(Z::Array; Factors = length(T.Values), inverse = false)\n\nCalling a PCA object on new data brings the new data Z into or out of (inverse = true) the PCA basis.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PartialLeastSquares-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PartialLeastSquares",
    "category": "method",
    "text": "PartialLeastSquares( X, Y; Factors = minimum(size(X)) - 2, tolerance = 1e-8, maxiters = 200 )\n\nReturns a PartialLeastSquares regression model object from arrays X and Y.\n\nPARTIAL LEAST-SQUARES REGRESSION: A TUTORIAL PAUL GELADI and BRUCE R.KOWALSKI. Analytica Chimica Acta, 186, (1986) PARTIAL LEAST-SQUARES REGRESSION:\nMartens H., NÊs T. Multivariate Calibration. Wiley: New York, 1989.\nRe-interpretation of NIPALS results solves PLSR inconsistency problem. Rolf Ergon. Published in Journal of Chemometrics 2009; Vol. 23/1: 72-75\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PartialLeastSquares-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PartialLeastSquares",
    "category": "method",
    "text": "(M::PartialLeastSquares)\n\nMakes an inference from X using a PartialLeastSquares object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Particle-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Particle",
    "category": "method",
    "text": "Particle(ProblemBounds, VelocityBounds)\n\nDefault constructor for a Particle object. It creates a random unformly distributed particle within the specified ProblemBounds, and limits it\'s velocity to the specified VelocityBounds.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PrincipalComponentRegression-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PrincipalComponentRegression",
    "category": "method",
    "text": "(M::PrincipalComponentRegression)( X )\n\nMakes an inference from X using a PrincipalComponentRegression object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PrincipalComponentRegression-Tuple{PCA,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PrincipalComponentRegression",
    "category": "method",
    "text": "PrincipalComponentRegression(PCAObject, Y )\n\nMakes a PrincipalComponentRegression model object from a PCA Object and property value Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.QQ-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.QQ",
    "category": "method",
    "text": "QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )\n\nReturns a Plot object of a Quantile-Quantile plot between vectors Y1 and Y2 at the desired Quantiles.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RandomForest",
    "page": "Documentation",
    "title": "ChemometricsTools.RandomForest",
    "category": "type",
    "text": "RandomForest(x, y, mode = :classification; gainfn = entropy, trees = 50, maxdepth = 10,  minbranchsize = 5, samples = 0.7, maxvars = nothing)\n\nReturns a classification (mode = :classification) or a regression (mode = :regression) random forest model. The gainfn can be entropy or gini for classification or ssd for regression. If the number of maximumvars is not provided it will default to sqrt(variables) for classification or variables/3 for regression.\n\nThe returned object can be used for inference by calling new data on the object as a function.\n\nBreiman, L. Machine Learning (2001) 45: 5. https://doi.org/10.1023/A:1010933404324\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RandomForest-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RandomForest",
    "category": "method",
    "text": "(RF::RandomForest)(X)\n\nReturns bagged prediction vector of random forest model.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RidgeRegression-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RidgeRegression",
    "category": "method",
    "text": "RidgeRegression( X, Y, Penalty; Bias = false )\n\nMakes a RidgeRegression model of the form Y = AX with or without a Bias term and has an L2 Penalty. Returns a CLS object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RidgeRegression-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RidgeRegression",
    "category": "method",
    "text": "(M::RidgeRegression)(X)\n\nMakes an inference from X using a RidgeRegression object which wraps a ClassicLeastSquares object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RollingWindow-Tuple{Int64,Int64,Int64}",
    "page": "Documentation",
    "title": "ChemometricsTools.RollingWindow",
    "category": "method",
    "text": "RollingWindow(samples::Int,windowsize::Int,skip::Int)\n\nCreates a RollingWindow iterator from a number of samples and a static windowsize where every iteration skip steps are skipped. The iterator can be used in for loops to iteratively return indices of a dynamic rolling window.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RollingWindow-Tuple{Int64,Int64}",
    "page": "Documentation",
    "title": "ChemometricsTools.RollingWindow",
    "category": "method",
    "text": "RollingWindow(samples::Int,windowsize::Int)\n\nCreates a RollingWindow iterator from a number of samples and a static windowsize. The iterator can be used in for loops to iteratively return indices of a dynamic rolling window.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RunningMean-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RunningMean",
    "category": "method",
    "text": "RunningMean(x)\n\nConstructs a running mean object with an initial scalar value of x.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RunningVar-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RunningVar",
    "category": "method",
    "text": "RunningVar(x)\n\nConstructs a RunningVar object with an initial scalar value of x. Note: RunningVar objects implicitly calculate the running mean.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.TransferByOrthogonalProjection-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.TransferByOrthogonalProjection",
    "category": "method",
    "text": "TransferByOrthogonalProjection(X1, X2; Factors = 1)\n\nPerforms Thomas Fearns Transfer By Orthogonal Projection to facilitate transfer from X1 to X2. Returns a TransferByOrthogonalProjection object.\n\nAnne Andrew, Tom Fearn. Transfer by orthogonal projection: making near-infrared calibrations robust to between-instrument variation. Chemometrics and Intelligent Laboratory Systems. Volume 72, Issue 1, 2004, Pages 51-56,\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.TransferByOrthogonalProjection-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.TransferByOrthogonalProjection",
    "category": "method",
    "text": "(TbOP::TransferByOrthogonalProjection)(X1; Factors = TbOP.Factors)\n\nApplies a the transform from a learned transfer by orthogonal projection object TbOP to new data X1.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ALSSmoother-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ALSSmoother",
    "category": "method",
    "text": "ALSSmoother(y; lambda = 100, p = 0.001, maxiters = 10)\n\nApplies an assymetric least squares smoothing function to a vector y. The lambda, p, and maxiters parameters control the smoothness. See the reference below for more information.\n\nPaul H. C. Eilers, Hans F.M. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing.  2005\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.BTEM",
    "page": "Documentation",
    "title": "ChemometricsTools.BTEM",
    "category": "function",
    "text": "BTEM(X, bands = nothing; Factors = 3, particles = 50, maxiters = 1000)\n\nReturns a single recovered spectra from a 2-Array X, the selected bands, number of Factors, using a Particle Swarm Optimizer.\n\nNote: This is not the function used in the original paper. This will be updated... it was written from memory. Also the original method uses Simulated Annealing not PSO. Band-Target Entropy Minimization (BTEM):  An Advanced Method for Recovering Unknown Pure Component Spectra. Application to the FTIR Spectra of Unstable Organometallic Mixtures. Wee Chew,Effendi Widjaja, and, and Marc Garland. Organometallics 2002 21 (9), 1982-1990. DOI: 10.1021/om0108752\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.BTEMobjective-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.BTEMobjective",
    "category": "method",
    "text": "BTEMobjective( a, X )\n\nReturns the scalar BTEM objective function obtained from the linear combination vector a and loadings X.\n\nNote: This is not the function used in the original paper. This will be updated... it was written from memory.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.BetweenClusterSS-Tuple{ChemometricsTools.ClusterModel}",
    "page": "Documentation",
    "title": "ChemometricsTools.BetweenClusterSS",
    "category": "method",
    "text": "BetweenClusterSS( Clustered::ClusterModel )\n\nReturns a scalar of the between cluster sum of squares for a ClusterModel object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ClassificationTree-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ClassificationTree",
    "category": "method",
    "text": "ClassificationTree(x, y; gainfn = entropy, maxdepth = 4, minbranchsize = 3)\n\nBuilds a CART object using either gini or entropy as a partioning method. Y must be a one hot encoded 2-Array. Predictions can be formed by calling the following function from the CART object: (M::CART)(x).\n\n*Note: this is a purely nonrecursive decision tree. The julia compiler doesn\'t like storing structs of nested things. I wrote it the recursive way in the past and it was quite slow, I think this is true also of interpretted languages like R/Python...So here it is, nonrecursive tree\'s!\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.DirectStandardization-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.DirectStandardization",
    "category": "method",
    "text": "DirectStandardization(InstrumentX1, InstrumentX2; Factors = minimum(collect(size(InstrumentX1))) - 1)\n\nMakes a DirectStandardization object to facilitate the transfer from Instrument #2 to Instrument #1 . The returned object can be used to transfer unseen data to the approximated space of instrument 1. The number of Factors used are those from the internal orthogonal basis.\n\nYongdong Wang and Bruce R. Kowalski, \"Calibration Transfer and Measurement Stability of Near-Infrared Spectrometers,\" Appl. Spectrosc. 46, 764-771 (1992)\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.EWMA-Tuple{Array,Float64}",
    "page": "Documentation",
    "title": "ChemometricsTools.EWMA",
    "category": "method",
    "text": "EWMA(Initial::Float64, Lambda::Float64) = ewma(Lambda, Initial, Initial, RunningVar(Initial))\n\nConstructs an exponentially weighted moving average object from an vector of scalar property values Initial and the decay parameter Lambda. This computes the running statistcs neccesary for creating the EWMA model using the interval provided and updates the center value to the mean of the provided values.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.EWMA-Tuple{Float64,Float64}",
    "page": "Documentation",
    "title": "ChemometricsTools.EWMA",
    "category": "method",
    "text": "EWMA(Initial::Float64, Lambda::Float64) = ewma(Lambda, Initial, Initial, RunningVar(Initial))\n\nConstructs an exponentially weighted moving average object from an initial scalar property value Initial and the decay parameter Lambda. This defaults the center value to be the initial value.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.EmpiricalQuantiles-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.EmpiricalQuantiles",
    "category": "method",
    "text": "EmpiricalQuantiles(X, quantiles)\n\nFinds the column-wise quantiles of 2-Array X and returns them in a 2-Array of size quantiles by variables. *Note: This copies the array... Use a subset if memory is the concern. *\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.EuclideanDistance-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.EuclideanDistance",
    "category": "method",
    "text": "EuclideanDistance(X, Y)\n\nReturns the euclidean distance matrix of X and Y such that the columns are the samples in Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.EuclideanDistance-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.EuclideanDistance",
    "category": "method",
    "text": "EuclideanDistance(X)\n\nReturns the Grahm aka the euclidean distance matrix of X.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ExplainedVariance-Tuple{LDA}",
    "page": "Documentation",
    "title": "ChemometricsTools.ExplainedVariance",
    "category": "method",
    "text": "ExplainedVariance(lda::LDA)\n\nCalculates the explained variance of each singular value in an LDA object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ExplainedVariance-Tuple{PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.ExplainedVariance",
    "category": "method",
    "text": "ExplainedVariance(PCA::PCA)\n\nCalculates the explained variance of each singular value in a pca object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ExtremeLearningMachine",
    "page": "Documentation",
    "title": "ChemometricsTools.ExtremeLearningMachine",
    "category": "function",
    "text": "ExtremeLearningMachine(X, Y, ReservoirSize = 10; ActivationFn = sigmoid)\n\nReturns a ELM regression model object from arrays X and Y, with a user specified ReservoirSize and ActivationFn.\n\nExtreme learning machine: a new learning scheme of feedforward neural networks. Guang-Bin Huang ; Qin-Yu Zhu ; Chee-Kheong Siew. 	2004 IEEE International Joint...\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.FNNLS-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.FNNLS",
    "category": "method",
    "text": "FNNLS(A, b; LHS = false, maxiters = 520)\n\nUses an implementation of Bro et. al\'s Fast Non-Negative Least Squares on the matrix A and vector b. We can state whether to pose the problem has a left-hand side problem (LHS = true) or a right hand side problem (default). Returns regression coefficients in the form of a vector.\n\nNote: this function does not have guarantees. Use at your own risk for now. Fast Non-Negative Least Squares algorithm based on Bro, R., & de Jong, S. (1997) A fast non-negativity-constrained least squares algorithm. Journal of Chemometrics, 11, 393-401.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.FirstDerivative-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.FirstDerivative",
    "category": "method",
    "text": "FirstDerivative(X)\n\nUses the finite difference method to compute the first derivative for every row in X. Note: This operation results in the loss of a column dimension.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.FractionalDerivative",
    "page": "Documentation",
    "title": "ChemometricsTools.FractionalDerivative",
    "category": "function",
    "text": "FractionalDerivative(Y, X = 1 : length(Y); Order = 0.5)\n\nCalculates the Grunwald-Leitnikov fractional order derivative on every row of Array Y. Array X is a vector that has the spacing between column-wise entries in Y. X can be a scalar if that is constant (common in spectroscopy). Order is the fractional order of the derivative. Note: This operation results in the loss of a column dimension.\n\nThe Fractional Calculus, by Oldham, K.; and Spanier, J. Hardcover: 234 pages. Publisher: Academic Press, 1974. ISBN 0-12-525550-0\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianKernel-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianKernel",
    "category": "method",
    "text": "GaussianKernel(X, Y, sigma)\n\nCreates a Gaussian/RBF kernel from Arrays X and Y with hyperparameter sigma.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.GaussianKernel-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.GaussianKernel",
    "category": "method",
    "text": "GaussianKernel(X, sigma)\n\nCreates a Gaussian/RBF kernel from Array X using hyperparameter sigma.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Hotelling-Tuple{Any,PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.Hotelling",
    "category": "method",
    "text": "Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)\n\nComputes the hotelling Tsq and upper control limit cut off of a pca object using a specified Quantile and cumulative variance explained Variance for new or old data X.\n\nA review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.IntervalOverlay-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.IntervalOverlay",
    "category": "method",
    "text": "IntervalOverlay(Spectra, Intervals, Err)\n\nDisplays the relative error(Err) of each interval ontop of a Spectra.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KMeans-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KMeans",
    "category": "method",
    "text": "KMeans( X, Clusters; tolerance = 1e-8, maxiters = 200 )\n\nReturns a ClusterModel object after finding clusterings for data in X via MacQueens K-Means algorithm. Clusters is the K parameter, or the # of clusters.\n\nMacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations. Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability. 1. University of California Press. pp. 281–297.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KennardStone-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KennardStone",
    "category": "method",
    "text": "KennardStone(X, TrainSamples; distance = \"euclidean\")\n\nReturns the indices of the Kennard-Stone sampled exemplars (E), and those not sampled (O) as a 2-Tuple (E, O).\n\nR. W. Kennard & L. A. Stone (1969) Computer Aided Design of Experiments, Technometrics, 111, 137-148, DOI: 10.1080/00401706.1969.10490666\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KernelRidgeRegression-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KernelRidgeRegression",
    "category": "method",
    "text": "KernelRidgeRegression( X, Y, Penalty; KernelParameter = 0.0, KernelType = \"linear\" )\n\nMakes a KernelRidgeRegression model of the form Y = AK using a user specified Kernel(\"Linear\", or \"Guassian\") and has an L2 Penalty. Returns a KRR Wrapper for a CLS object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Leverage-Tuple{PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.Leverage",
    "category": "method",
    "text": "Leverage(pca::PCA)\n\nCalculates the leverage of samples in a pca object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Limits-Tuple{ChemometricsTools.ewma}",
    "page": "Documentation",
    "title": "ChemometricsTools.Limits",
    "category": "method",
    "text": "Limits(P::ewma; k = 3.0)\n\nThis function returns the upper and lower control limits with a k span of variance for an EWMA object P. \n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LinearKernel-Tuple{Any,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LinearKernel",
    "category": "method",
    "text": "LinearKernel(X, Y, c)\n\nCreates a Linear kernel from Arrays X and Y with hyperparameter C.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.LinearKernel-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.LinearKernel",
    "category": "method",
    "text": "LinearKernel(X, c)\n\nCreates a Linear kernel from Array X and hyperparameter C.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MAE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.MAE",
    "category": "method",
    "text": "MAE( y, yhat )\n\nCalculates Mean Average Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MAPE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.MAPE",
    "category": "method",
    "text": "MAPE( y, yhat )\n\nCalculates Mean Average Percent Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MCRALS",
    "page": "Documentation",
    "title": "ChemometricsTools.MCRALS",
    "category": "function",
    "text": "MCRALS(X, C, S = nothing; norm = (false, false), Factors = 1, maxiters = 20, nonnegative = (false, false) )\n\nPerforms Multivariate Curve Resolution using Alternating Least Squares on X taking initial estimates for S or C. S or C can be constrained by their norm, or by nonnegativity using nonnegative arguments. The number of resolved Factors can also be set.\n\nTauler, R. Izquierdo-Ridorsa, A. Casassas, E. Simultaneous analysis of several spectroscopic titrations with self-modelling curve resolution.Chemometrics and Intelligent Laboratory Systems. 18, 3, (1993), 293-300.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ME-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ME",
    "category": "method",
    "text": "ME( y, yhat )\n\nCalculates Mean Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MSE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.MSE",
    "category": "method",
    "text": "MSE( y, yhat )\n\nCalculates Mean Squared Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MakeIntervals",
    "page": "Documentation",
    "title": "ChemometricsTools.MakeIntervals",
    "category": "function",
    "text": "MakeIntervals( columns::Int, intervalsize::Union{Array, Tuple} = [20, 50, 100] )\n\nCreates an Dictionary whose key is the interval size and values are an array of intervals from the range: 1 - columns of size intervalsize.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MakeIntervals",
    "page": "Documentation",
    "title": "ChemometricsTools.MakeIntervals",
    "category": "function",
    "text": "MakeIntervals( columns::Int, intervalsize::Int = 20 )\n\nReturns an 1-Array of intervals from the range: 1 - columns of size intervalsize.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ManhattanDistance-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ManhattanDistance",
    "category": "method",
    "text": "ManhattanDistance(X, Y)\n\nReturns the Manhattan distance matrix of X and Y such that the columns are the samples in Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ManhattanDistance-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ManhattanDistance",
    "category": "method",
    "text": "ManhattanDistance(X)\n\nReturns the Manhattan distance matrix of X.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Mean-Tuple{RunningMean}",
    "page": "Documentation",
    "title": "ChemometricsTools.Mean",
    "category": "method",
    "text": "Mean(rv::RunningMean)\n\nReturns the current mean inside of a RunningMean object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Mean-Tuple{RunningVar}",
    "page": "Documentation",
    "title": "ChemometricsTools.Mean",
    "category": "method",
    "text": "Mean(rv::RunningVar)\n\nReturns the current mean inside of a RunningVar object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.MultinomialSoftmaxRegression-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.MultinomialSoftmaxRegression",
    "category": "method",
    "text": "MultinomialSoftmaxRegression(X, Y; LearnRate = 1e-3, maxiters = 1000, L2 = 0.0)\n\nReturns a LogisticRegression classification model made by Stochastic Gradient Descent.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.NMF-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.NMF",
    "category": "method",
    "text": "NMF(X; Factors = 1, tolerance = 1e-7, maxiters = 200)\n\nPerforms a variation of non-negative matrix factorization on Array X and returns the a 2-Tuple of (Concentration Profile, Spectra)\n\nNote: This is not a coordinate descent based NMF. This is a simple fast version which works well enough for chemical signals Algorithms for non-negative matrix factorization. Daniel D. Lee. H. Sebastian Seung. NIPS\'00 Proceedings of the 13th International Conference on Neural Information Processing Systems. 535-54\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.OneHotOdds-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.OneHotOdds",
    "category": "method",
    "text": "OneHotOdds(Y)\n\nCalculates the odds of a one-hot formatted probability matrix. Returns a tuple.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PCA_NIPALS-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PCA_NIPALS",
    "category": "method",
    "text": "PCA_NIPALS(X; Factors = minimum(size(X)) - 1, tolerance = 1e-7, maxiters = 200)\n\nCompute\'s a PCA from x using the NIPALS algorithm with a user specified number of latent variables(Factors). The tolerance is the minimum change in the F norm before ceasing execution. Returns a PCA object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PSO-Tuple{Any,Bounds,Bounds,Int64}",
    "page": "Documentation",
    "title": "ChemometricsTools.PSO",
    "category": "method",
    "text": "PSO(fn, Bounds, VelRange, Particles; tolerance = 1e-6, maxiters = 1000, InertialDecay = 0.5, PersonalWeight = 0.5, GlobalWeight = 0.5, InternalParams = nothing)\n\nMinimizes function fn with-in the user specified Bounds via a Particle Swarm Optimizer. The particle velocities are limitted to the VelRange. The number of particles are defined by the Particles parameter.\n\nReturns a Tuple of the following form: ( GlobalBestPos, GlobalBestScore, P ) Where P is an array of the particles used in the optimization.\n\n*Note: if the optimization function requires an additional constant parameter, please pass that parameter to InternalParams. This will only work if the optimized parameter(o) and constant parameter(c) for the function of interest has the following format: F(o,c) *\n\nKennedy, J.; Eberhart, R. (1995). Particle Swarm Optimization. Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942–1948. doi:10.1109/ICNN.1995.488968\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PearsonCorrelationCoefficient-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PearsonCorrelationCoefficient",
    "category": "method",
    "text": "PearsonCorrelationCoefficient( y, yhat )\n\nCalculates The Pearson Correlation Coefficient from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PercentRMSE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PercentRMSE",
    "category": "method",
    "text": "PercentRMSE( y, yhat )\n\nCalculates Percent Root Mean Squared Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.PerfectSmoother-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.PerfectSmoother",
    "category": "method",
    "text": "PerfectSmoother(y; lambda = 100)\n\nApplies an assymetric least squares smoothing function to a vector y. The lambda parameter controls the smoothness. See the reference below for more information.\n\nPaul H. C. Eilers. \"A Perfect Smoother\". Analytical Chemistry, 2003, 75 (14), pp 3631–3636.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Q-Tuple{Any,PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.Q",
    "category": "method",
    "text": "Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)\n\nComputes the Q-statistic and upper control limit cut off of a pca object using a specified Quantile and cumulative variance explained Variance for new or old data X.\n\nA review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RMSE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RMSE",
    "category": "method",
    "text": "RMSE( y, yhat )\n\nCalculates Root Mean Squared Error from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.RSquare-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.RSquare",
    "category": "method",
    "text": "RSquare( y, yhat )\n\nCalculates R^2 from Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Remove!-Tuple{RunningMean,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Remove!",
    "category": "method",
    "text": "Remove!(RM::RunningMean, x)\n\nRemoves an observation(x) from a RunningMean object(RM) and reculates the mean in place.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Remove-Tuple{RunningMean,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Remove",
    "category": "method",
    "text": "Remove!(RM::RunningMean, x)\n\nRemoves an observation(x) from a RunningMean object(RM) and recuturns the new RunningMean object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SIMPLISMA-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SIMPLISMA",
    "category": "method",
    "text": "SIMPLISMA(X; Factors = 1)\n\nPerforms SIMPLISMA on Array X. Returns a tuple of the following form: (Concentraion Profile, Pure Spectral Estimates, Pure Variables)\n\nNote: This is not the traditional SIMPLISMA algorithm presented by Willem Windig. REAL-TIME WAVELET COMPRESSION AND SELF-MODELING CURVE RESOLUTION FOR ION MOBILITY SPECTROMETRY. PhD. Dissertation. 2003. Guoxiang Chen.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SSE-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SSE",
    "category": "method",
    "text": "SSE( y, yhat )\n\nCalculates Sum of Squared Errors from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SSReg-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SSReg",
    "category": "method",
    "text": "SSReg( y, yhat )\n\nCalculates Sum of Squared Deviations due to Regression from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SSRes-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SSRes",
    "category": "method",
    "text": "SSRes( y, yhat )\n\nCalculates Sum of Squared Residuals from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SSTotal-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SSTotal",
    "category": "method",
    "text": "SSTotal( y, yhat )\n\nCalculates Total Sum of Squared Deviations from vectors Y and YHat\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SavitzkyGolay-NTuple{4,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SavitzkyGolay",
    "category": "method",
    "text": "SavitzkyGolay(X, Delta, PolyOrder, windowsize)\n\nPerforms SavitskyGolay smoothing across every row in an Array X. The window size is the size of the convolution filter, PolyOrder is the order of the polynomial, and Delta is the order of the derivative.\n\nSavitzky, A.; Golay, M.J.E. (1964). \"Smoothing and Differentiation of Data by Simplified Least Squares Procedures\". Analytical Chemistry. 36 (8): 1627–39. doi:10.1021/ac60214a047.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Scale1Norm-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Scale1Norm",
    "category": "method",
    "text": "Scale1Norm(X)\n\nScales the columns of X by the 1-Norm of each row. Returns the scaled array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Scale2Norm-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Scale2Norm",
    "category": "method",
    "text": "Scale2Norm(X)\n\nScales the columns of X by the 2-Norm of each row. Returns the scaled array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ScaleInfNorm-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ScaleInfNorm",
    "category": "method",
    "text": "ScaleInfNorm(X)\n\nScales the columns of X by the Inf-Norm of each row. Returns the scaled array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SecondDerivative-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SecondDerivative",
    "category": "method",
    "text": "FirstDerivative(X)\n\nUses the finite difference method to compute the second derivative for every row in X. Note: This operation results in the loss of two columns.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Shuffle!-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Shuffle!",
    "category": "method",
    "text": "Shuffle!( X, Y )\n\nShuffles the rows of the X and Y data without replacement in place. In place, means that this function alters the order of the data in memory and this function does not return anything.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Shuffle-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Shuffle",
    "category": "method",
    "text": "Shuffle( X, Y )\n\nShuffles the rows of the X and Y data without replacement. It returns a 2-Tuple of the shuffled set.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SplitByProportion",
    "page": "Documentation",
    "title": "ChemometricsTools.SplitByProportion",
    "category": "function",
    "text": "SplitByProportion(X::Array, Proportion::Float64 = 0.5)\n\nSplits X Array along the observations dimension into a 2-Tuple based on the Proportion. The form of the output is the following: ( X1, X2 )\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SplitByProportion",
    "page": "Documentation",
    "title": "ChemometricsTools.SplitByProportion",
    "category": "function",
    "text": "SplitByProportion(X::Array, Y::Array,Proportion::Float64 = 0.5)\n\nSplits an X and Associated Y Array along the observations dimension into a 2-Tuple of 2-Tuples based on the Proportion. The form of the output is the following: ( (X1, Y1), (X2, Y2) )\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SquareEuclideanDistance-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SquareEuclideanDistance",
    "category": "method",
    "text": "SquareEuclideanDistance(X, Y)\n\nReturns the squared euclidean distance matrix of X and Y such that the columns are the samples in Y.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.SquareEuclideanDistance-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.SquareEuclideanDistance",
    "category": "method",
    "text": "SquareEuclideanDistance(X)\n\nReturns the squared Grahm aka the euclidean distance matrix of X.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.StandardNormalVariate-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.StandardNormalVariate",
    "category": "method",
    "text": "StandardNormalVariate(X)\n\nScales the columns of X by the mean and standard deviation of each row. Returns the scaled array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.TotalClusterSS-Tuple{ChemometricsTools.ClusterModel}",
    "page": "Documentation",
    "title": "ChemometricsTools.TotalClusterSS",
    "category": "method",
    "text": "TotalClusterSS( Clustered::ClusterModel )\n\nReturns a scalar of the total sum of squares for a ClusterModel object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Update!-Tuple{RunningMean,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Update!",
    "category": "method",
    "text": "Update!(RM::RunningMean, x)\n\nAdds new observation(x) to a RunningMean object(RM) in place.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Update!-Tuple{RunningVar,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Update!",
    "category": "method",
    "text": "Update!(RV::RunningVar, x)\n\nAdds new observation(x) to a RunningVar object(RV) and updates it in place.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Update-Tuple{RunningMean,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.Update",
    "category": "method",
    "text": "Update!(RM::RunningMean, x)\n\nAdds new observation(x) to a RunningMean object(RM) and returns the new object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Variance-Tuple{ChemometricsTools.ewma}",
    "page": "Documentation",
    "title": "ChemometricsTools.Variance",
    "category": "method",
    "text": "Variance(P::ewma)\n\nThis function returns the EWMA control variance.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.Variance-Tuple{RunningVar}",
    "page": "Documentation",
    "title": "ChemometricsTools.Variance",
    "category": "method",
    "text": "Variance(rv::RunningVar)\n\nReturns the current variance inside of a RunningVar object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.VenetianBlinds-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.VenetianBlinds",
    "category": "method",
    "text": "VenetianBlinds(X,Y)\n\nSplits an X and associated Y Array along the observation dimension into a 2-Tuple of 2-Tuples based on the whether it is even or odd. The form of the output is the following: ( (X1,Y1), (X2, Y2) )\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.VenetianBlinds-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.VenetianBlinds",
    "category": "method",
    "text": "VenetianBlinds(X)\n\nSplits an X Array along the observations dimension into a 2-Tuple of 2-Tuples based on the whether it is even or odd. The form of the output is the following: ( X1, X2 )\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.WithinClusterSS-Tuple{ChemometricsTools.ClusterModel}",
    "page": "Documentation",
    "title": "ChemometricsTools.WithinClusterSS",
    "category": "method",
    "text": "WithinClusterSS( Clustered::ClusterModel )\n\nReturns a scalar of the within cluter sum of squares for a ClusterModel object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.entropy-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.entropy",
    "category": "method",
    "text": "entropy(v)\n\nCalculates the Shannon-Entropy of a probability vector v. Returns a scalar. A common gain function used in tree methods.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.findpeaks-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.findpeaks",
    "category": "method",
    "text": "findpeaks( vY; m = 3)\n\nFinds the indices of peaks in a vector vY with a window span of 2m. Original R function by Stas_G:(https://stats.stackexchange.com/questions/22974/how-to-find-local-peaks-valleys-in-a-series-of-data) This version is based on a C++ variant by me.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.gini-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.gini",
    "category": "method",
    "text": "gini(p)\n\nCalculates the GINI coefficient of a probability vector p. Returns a scalar. A common gain function used in tree methods.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.offsetToZero-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.offsetToZero",
    "category": "method",
    "text": "offsetToZero(X)\n\nEnsures that no observation(row) of Array X is less than zero, by ensuring the minimum value of each row is zero.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.plotchem-Tuple{QQ}",
    "page": "Documentation",
    "title": "ChemometricsTools.plotchem",
    "category": "method",
    "text": "plotchem(QQ::{QQ, BlandAltman}; title )\n\nreturns either a QQ Plot or a Bland-Altman plot with the defined title\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.rbinomial-Tuple{Any,Vararg{Any,N} where N}",
    "page": "Documentation",
    "title": "ChemometricsTools.rbinomial",
    "category": "method",
    "text": "rbinomial( p, size... )\n\nMakes an N-dimensional array of size(s) size with a probability of being a 1 over a 0 of 1 p.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.sigmoid-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.sigmoid",
    "category": "method",
    "text": "sigmoid(x)\n\nApplies the sigmoid function to a scalar value X. Returns a scalar. Can be broad-casted over an Array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ssd-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ssd",
    "category": "method",
    "text": "ssd(p)\n\nCalculates the sum squared deviations from a decision tree split. Accepts a vector of values, and the mean of that  vector. Returns a scalar. A common gain function used in tree methods.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.stackedweights-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.stackedweights",
    "category": "method",
    "text": "stackedweights(ErrVec; power = 2)\n\nWeights stacked interval errors by the reciprocal power specified. Used for SIPLS, SISPLS, etc.\n\nNi, W. , Brown, S. D. and Man, R. (2009), Stacked partial least squares regression analysis for spectral calibration and prediction. J. Chemometrics, 23: 505-517. doi:10.1002/cem.1246\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.DirectStandardizationXform-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.DirectStandardizationXform",
    "category": "method",
    "text": "(DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.Values))\n\nApplies a the transform from a learned direct standardization object DSX to new data X.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ELM-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ELM",
    "category": "method",
    "text": "(M::ELM)(X)\n\nMakes an inference from X using a ELM object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.KRR-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KRR",
    "category": "method",
    "text": "(M::KRR)(X)\n\nMakes an inference from X using a KRR object which wraps a ClassicLeastSquares object.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ewma-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ewma",
    "category": "method",
    "text": "EWMA(P::ewma)(New; train = true)\n\nProvides an EWMA score for a New scalar value. If train == true the model is updated to include this new value.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ChangeCenter-Tuple{ChemometricsTools.ewma,Float64}",
    "page": "Documentation",
    "title": "ChemometricsTools.ChangeCenter",
    "category": "method",
    "text": "ChangeCenter(P::ewma, new::Float64)\n\nThis is a convenience function to update the center of a P EWMA model, to a new scalar value.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ScaleMinMax-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ScaleMinMax",
    "category": "method",
    "text": "ScaleMinMax(X)\n\nScales the columns of X by the Min and Max of each row such that no observation is greater than 1 or less than zero. Returns the scaled array.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.boxcar-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.boxcar",
    "category": "method",
    "text": "boxcar(X; windowsize = 3, fn = mean)\n\nApplies a boxcar function (fn) to each window of size windowsize to every row in X.\n\n\n\n\n\n"
},

{
    "location": "#Documentation-1",
    "page": "Documentation",
    "title": "Documentation",
    "category": "section",
    "text": "CurrentModule = ChemometricsTools\nDocTestSetup = quote\n	using ChemometricsTools\nendModules = [ChemometricsTools]"
},

]}
