var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Documentation",
    "title": "Documentation",
    "category": "page",
    "text": ""
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
    "location": "#ChemometricsTools.StandardNormalVariate-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.StandardNormalVariate",
    "category": "method",
    "text": "StandardNormalVariate(X)\n\nScales the columns of X by the mean and standard deviation of each row. Returns the scaled array.\n\n\n\n\n\n"
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
    "location": "#ChemometricsTools.rbinomial-Tuple{Any,Vararg{Any,N} where N}",
    "page": "Documentation",
    "title": "ChemometricsTools.rbinomial",
    "category": "method",
    "text": "rbinomial( p, size... )\n\nMakes an N-dimensional array of size(s) size with a probability of being a 1 over a 0 of 1 p.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.ssd-Tuple{Any,Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.ssd",
    "category": "method",
    "text": "ssd(p)\n\nCalculates the sum squared deviations from a decision tree split. Accepts a vector of values, and the mean of that  vector. Returns a scalar. A common gain function used in tree methods.\n\n\n\n\n\n"
},

{
    "location": "#ChemometricsTools.DirectStandardizationXform-Tuple{Any}",
    "page": "Documentation",
    "title": "ChemometricsTools.DirectStandardizationXform",
    "category": "method",
    "text": "(DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.Values))\n\nApplies a the transform from a learned direct standardization object DSX to new data X.\n\n\n\n\n\n"
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
