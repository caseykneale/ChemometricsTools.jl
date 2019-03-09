[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://caseykneale.github.io/ChemometricsTools.jl/) [![Build Status](https://travis-ci.org/caseykneale/ChemometricsTools.jl.svg?branch=master)](https://travis-ci.org/caseykneale/ChemometricsTools.jl)

# ChemometricsTools.jl
This package contains a collection of tools to perform fundamental and advanced Chemometric analysis' in Julia. It is currently richer and more fundamental then any single free chemometrics package available in any other language. If you are uninformed as to what Chemometrics is; it could nonelegantly be described as the marriage between data science and chemistry. Traditionally it is a pile of applied linear algebra/statistics that is well reasoned by the physics and meaning of chemical measurements. This is somewhat orthogonal to most fields of machine learning (aka "add more layers"). Sometimes chemometricians also get desperate and break out pure machine learning methods. So some of those methods are in this package, but if you want neural networks try [Flux.jl](https://github.com/FluxML/Flux.jl).

## Tutorials/Demonstrations:
  - [Transforms](https://caseykneale.github.io/ChemometricsTools.jl/Demos/Transforms/)
  - [Pipelines](https://caseykneale.github.io/ChemometricsTools.jl/Demos/Pipelines/)
  - [Classification](https://caseykneale.github.io/ChemometricsTools.jl/Demos/ClassificationExample/)
  - [Regression](https://caseykneale.github.io/ChemometricsTools.jl/Demos/RegressionExample/)
  - [Calibration Transfer: Direct Standardization](https://caseykneale.github.io/ChemometricsTools.jl/Demos/CalibXfer/)
  - [Stacked Interval Partial Least Squares Regression](https://caseykneale.github.io/ChemometricsTools.jl/Demos/SIPLS/)
  - [Curve Resolution](https://caseykneale.github.io/ChemometricsTools.jl/Demos/CurveResolution/)

## Shootouts/Modeling Examples:
  - [Readme](https://github.com/caseykneale/ChemometricsTools.jl/tree/master/shootouts)
  - [Classification](https://github.com/caseykneale/ChemometricsTools.jl/blob/master/shootouts/ClassificationShootout.jl)
  - [Regression](https://github.com/caseykneale/ChemometricsTools.jl/blob/master/shootouts/RegressionShootout.jl)
  - [Fault Detection](https://github.com/caseykneale/ChemometricsTools.jl/blob/master/shootouts/AnomalyShootout.jl)

### Package Status => Public release! (v 0.2.3)
ChemometricsTools is brand new, and was recently accepted to be registered as an official Julia package! Yep, so you can  ```Pkg.add("ChemometricsTools")``` to install it. The git repo has the most stable version right now, I fixed a lot of bugs since 0.2.3, but soon the package will be up to date. Almost all of the functionality available can reliably be used/abused, and the documentation is pretty fleshed out. Betchya anything there are a few bugs in the package still. This is an early release for constructive criticism and spreading awareness.

# Package Highlights
### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". We can use transformations to treat data from multiple sources the same way. This helps mitigate user error for cases where test data is scaled based on training data, calibration transfer, etc.

Multiple transformations can easily be chained together and stored using "Pipelines". Pipelines aren't "pipes" like are present in Bash, R and base Julia. They are flexible, yet immutable, convenience objects that allow for sequential preprocessing and data transformations to be reused, chained, or automated for reliable analytic throughput.

### Model training
ChemometricsTools offers easy to use iterators for K-folds validation's, and moving window sampling. More advanced sampling methods, like Kennard Stone, are just a function call away. Convenience functions for interval selections, weighting regression ensembles, etc are also available. These allow for ensemble models like SIPLS, P-DS, P-OSC, etc to be built quickly.

### Regression Modeling
This package features dozens of regression performance metrics, and a few built in plots (Bland Altman, QQ, Interval Overlays etc) are included. The list of regression methods currently includes: CLS, Ridge, Kernel Ridge, LS-SVM, PCR, PLS(1/2), ELM's, Regression Trees, Random Forest... More to come. Chemometricians love regressions!

### Classification Modeling
In-house classification encodings (one cold/one hot), multiclass performance statistics. ChemometricsTools currently includes: LDA with Gaussian discriminants, logistic regression, PLS-DA, K-NN, Gaussian Naive Bayes, Classification Trees, Random Forest, and more to come.

## Specialized tools?
This package has tools for specialized fields of analysis'. For instance, fractional derivatives for the electrochemists (and the adventurous), Savitsky Golay smoothing for spectroscopists, curve resolution for forensics, process fault detection methods, etc. There are certainly plans for other tools for analyzing chemical data that packages in other languages have seemingly left out. Stay tuned.

## Wheres the Data?
Right now I don't have rights to any data; but iris, the Tecator meat data, and a NASA fault detection dataset are included. I'd love for a collaborator to contribute some: spectra, mass spectra, chromatograms, etc. Please reach out to me if you wish to collaborate/contribute. There's a good chance in a week or so I'll be reaching out to the community for these sorts of things, in the mean time you can load in your own datasets using the Julia ecosystem.

## What about Time Series? Cluster models?
Well, I'd love to hammer in some time series methods. That was originally part of the plan. Then I realized [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) already has pretty much everything covered. Similarly, if you want clustering methods, just install [Clustering.jl](https://github.com/JuliaStats/Clustering.jl). I'll add a few supportive odds and ends in here but really, some of the Julia ecosystem is really reliable and well made.

## ToDo:
  - SIMPLISMA return unique pure var's...
  - Double check FNNLS
  - Some more unit tests...
  - Long-term: SIMCA, N-WAY PCA, and N-WAY PLS

## Maybes:
  - Hyperspectral data preprocessing methods?
  - Convenience fns for standard addition, propagation of error, multiequilibria, kinetics?
  - Design of Experiment tools (Partial Factorial design, simplex, etc...)?
  - Electrochemical simulations and optical simulations?
