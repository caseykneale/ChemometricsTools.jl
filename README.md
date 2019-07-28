[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://caseykneale.github.io/ChemometricsTools.jl/) [![Build Status](https://travis-ci.org/caseykneale/ChemometricsTools.jl.svg?branch=master)](https://travis-ci.org/caseykneale/ChemometricsTools.jl)

# ChemometricsTools.jl
This package contains a collection of tools to perform fundamental and advanced Chemometric analysis' in Julia. It is currently richer and more fundamental than any single free chemometrics package available in any other language. If you are uninformed as to what Chemometrics is; it could nonelegantly be described as the marriage between data science and chemistry. Traditionally it is a pile of applied linear algebra/statistics that is well reasoned by the physics and meaning of chemical measurements. This is somewhat orthogonal to most fields of machine learning (aka "add more layers"). Sometimes chemometricians also get desperate and break out pure machine learning methods. So some of those methods are in this package, but if you want neural networks try [Flux.jl](https://github.com/FluxML/Flux.jl).

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

### Package Status => "Registrator" release! (v 0.4.6)
ChemometricsTools is pretty new, and was recently accepted to be registered as an official Julia package! Yep, so you can  ```Pkg.add("ChemometricsTools")``` to install it. The git repo's master branch has the most stable version right now, I fixed a lot of bugs since 0.2.3. In 0.4.6 almost all of the functionality available can reliably be used/abused, and the documentation is getting there, but it's hard to keep up with all the features I've been adding. There's probably still a few bugs. Some interesting plans for v0.5.0, but I've personally been testing this package doing some work with a fellow analytical chemist.

### Version Release Strategy
  - < 0.3.0 : Mapping functionality, prototyping
  - *< 0.5.0 : Testing via actual usage on real data, look for missing essentials*
  - < 0.7.5 : Public input (find those bugs!). Complete docs with examples. Adequate Unit Tests.
  - < 1.0.0 : Focus on performance, stability, generalizability, lock down the package syntax.

# Package Highlights
### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". We can use transformations to treat data from multiple sources the same way. This helps mitigate user error for cases where test data is scaled based on training data, calibration transfer, etc.

Multiple transformations can easily be chained together and stored using "Pipelines". Pipelines aren't "pipes" like are present in Bash, R and base Julia. They are flexible, yet immutable, convenience objects that allow for sequential preprocessing and data transformations to be reused, chained, or automated for reliable analytic throughput.

### Model training
ChemometricsTools offers easy to use iterators for K-folds validation's, and moving window sampling/training. More advanced sampling methods, like Kennard Stone, are just a function call away. Convenience functions for interval selections, weighting regression ensembles, etc are also available. These allow for ensemble models like SIPLS, P-DS, P-OSC, etc to be built quickly. With the tools included both in this package and Base Julia, nothing should stand in your way.

### Regression Modeling
This package features dozens of regression performance metrics, and a few built in plots (Bland Altman, QQ, Interval Overlays etc) are included. The list of regression methods currently includes: CLS, Ridge, Kernel Ridge, LS-SVM, PCR, PLS(1/2), ELM's, Regression Trees, Random Forest... More to come. Chemometricians love regressions!

### Classification Modeling
In-house classification encodings (one cold/one hot), and easy to retrieve global or multiclass performance statistics. ChemometricsTools currently includes: LDA/PCA with Gaussian discriminants, also Hierchical LDA, multinomial softmax/logistic regression, PLS-DA, K-NN, Gaussian Naive Bayes, Classification Trees, Random Forest, Probabilistic Neural Networks, LinearPerceptrons, and more to come.

## Specialized tools?
This package has tools for specialized fields of analysis'. For instance, fractional derivatives for the electrochemists (and the adventurous), a handful of smoothing methods for spectroscopists, curve resolution for forensics, process fault detection methods, etc. There are certainly plans for other tools for analyzing chemical data that packages in other languages have seemingly left out. Stay tuned.

## Where's the Data?
Right now I don't have rights to provide much data; but the iris, Tecator meat data, and a NASA fault detection datasets are included. I'd love for a collaborator to contribute some: spectra, chromatograms, etc. Please reach out to me if you wish to collaborate/contribute. There's a good chance in a week or so I'll be reaching out to the community for these sorts of things, in the mean time you can load in your own datasets using the Julia ecosystem.

## What about Time Series? Cluster models?
Well, I'd love to hammer in some time series methods. That was originally part of the plan. Then I realized [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) already has pretty much everything covered. Similarly, if you want clustering methods, just install [Clustering.jl](https://github.com/JuliaStats/Clustering.jl). I may add a few supportive odds and ends in here(or contribute to the packages directly) but really, most of the Julia 1.0+ ecosystem is really reliable, well made, and community supported.

## ToDo:
  - Add Unimodal constraint to MCR-ALS.
  - Make function to dump classification statistics to an autogenerated LaTeX report.
  - Gaussian Discriminant plotting function(needs cleaning and documenting)
  - Long-term: SIMCA, and MultiWAY PLS
  - Hyperspectral data preprocessing methods that fit into pipelines/transforms.

## Maybes:
  - Convenience fns for standard addition, propagation of error, multiequilibria, kinetics?
  - Design of Experiment tools (Partial Factorial design, simplex, etc...)?
  - Electrochemical simulations and optical simulations (maybe separate packages...)?
