var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Documentation",
    "title": "Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "#ChemometricsTools.KFoldsValidation-Tuple{Int64,Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.KFoldsValidation",
    "category": "method",
    "text": "KFoldsValidation(K::Int, x, y)\n\nReturns a KFoldsValidation iterator with K folds. Because it\'s an iterator it can be used in for loops, see the tutorials for pragmatic examples. The iterator returns a 2-Tuple of 2-Tuples which have the  following form: ((TrainX,TrainY),(ValidateX,ValidateY).\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.ClassificationTree-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ClassificationTree",
    "category": "method",
    "text": "ClassificationTree(x, y; gainfn = entropy, maxdepth = 4, minbranchsize = 3)\n\nBuilds a CART object using either gini or entropy as a partioning method. Y must be a one hot encoded 2-Array. Predictions can be formed by calling the following function from the CART object: (M::CART)(x).\n\n*Note: this is a purely nonrecursive decision tree. The julia compiler doesn\'t like storing structs of nested things. I wrote it the recursive way in the past and it was quite slow, I think this is true also of interpretted languages like R/Python...So here it is, nonrecursive tree\'s!\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.Hotelling-Tuple{Any,PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.Hotelling",
    "category": "method",
    "text": "Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)\n\nComputes the hotelling Tsq and upper control limit cut off of a pca object using a specified Quantile and cumulative variance explained Variance for new or old data X.\n\nA review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.Q-Tuple{Any,PCA}",
    "page": "Documentation",
    "title": "ChemometricsTools.Q",
    "category": "method",
    "text": "Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)\n\nComputes the Q-statistic and upper control limit cut off of a pca object using a specified Quantile and cumulative variance explained Variance for new or old data X.\n\nA review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.Variance-Tuple{ChemometricsTools.ewma}",
    "page": "Documentation",
    "title": "ChemometricsTools.Variance",
    "category": "method",
    "text": "Variance(P::ewma)\n\nThis function returns the EWMA control variance.\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.ssd-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ssd",
    "category": "method",
    "text": "ssd(p)\n\nCalculates the sum squared deviations from a decision tree split. Accepts a vector of values, and the mean of that  vector. Returns a scalar. A common gain function used in tree methods.\n\n\n\n\n\n"
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
    "location": "#Documentation-1",
    "page": "Documentation",
    "title": "Documentation",
    "category": "section",
    "text": "using Pkg\nPkg.activate(.)\nCurrentModule = ChemometricsTools\nDocTestSetup = quote\n	using ChemometricsTools\nendModules = [ChemometricsTools]"
},

]}
