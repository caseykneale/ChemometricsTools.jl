No official doc's yet, but the [wiki](https://github.com/caseykneale/ChemometricsTools/wiki) has the majority of the API and a few tutorials in it now!

# ChemometricsTools
This is an essential collection of tools to perform fundamental and advanced Chemometric analysis' in Julia. If you are uninformed as to what Chemometrics is; it could nonelegantly be described as the marriage between data science and chemistry. Traditionally it is a pile of applied linear algebra that is well reasoned by the physics and meaning of chemical measurements. This is some orthogonal to most fields of machine learning, but sometimes chemometricians also get desperate and break out pure machine learning methods. 

The goals for this package are the following: rely only on basic dependencies for longevity and stability, essential algorithms should read similar to pseudocode in papers, and the tools provided should be fast, flexible, and reliable. That's the Julia way after-all. No code will directly call R, Python, C, etc. As such it is currently and will be written in Julia from the ground up.

## Demonstrations:
  - [Calibration Transfer: Direct Standardization](https://github.com/caseykneale/ChemometricsTools/wiki/Calibration-Transfer:-Direct-Standardization-Demo)
  - [Curve Resolution](https://github.com/caseykneale/ChemometricsTools/wiki/Curve-Resolution:-Demo)
  - [Stacked Interval Partial Least Squares Regression](https://github.com/caseykneale/ChemometricsTools/wiki/Stacked-Interval-Partial-Least-Squares:-A-Demo)
## Doc's with Tutorials
  - [Classification](https://github.com/caseykneale/ChemometricsTools/wiki/Classification-Methods)
  - [Training](https://github.com/caseykneale/ChemometricsTools/wiki/Training-Methods)
  - [Transforms](https://github.com/caseykneale/ChemometricsTools/wiki/Transformations)
  - [Pipelines](https://github.com/caseykneale/ChemometricsTools/wiki/Pipelines)

## Ethos
Dependencies: Only base libraries (LinearAlgebra, StatsBase, Statistics, Plots) etc will be required. This is for longevity, and to provide a fast precompilation time. As wonderful as it is that other packages exist to do some of the internal operations this one needs, we won't have to worry about a breaking change made by an external author working out the kinks in a separate package. I want this to be long-term reliable even if I go MIA for a week or two.

Arrays Only: In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change.

Center-Scaling: None of the methods in this package will center and scale for you. This package won't waste your time by centering and scaling large chunks of data every-time you do a regression/classification.

### Package Status => Early View
This thing is brand new (< 3 weeks old). Many of the tools available can already be used, and most of those are implemented correctly, but the documentation is slow coming. Betchya anything there are a few bugs hiding in the repo. In a week or so this should be functional and trustworthy, and at that point collaborators will be sought. This is an early preview for constructive criticism and spreading awareness.

# Package Highlights
### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer, etc.

Multiple transformations can be easily chained together and stored using "Pipelines". These are basically convenience functions, but are somewhat flexible. Pipelines allow for preprocessing and data transformations to be reused, chained, or automated for reliable analytic throughput.

### Model training
Easy to use iterators for KFoldsValidation's! Sampling methods like Kennard Stone, and resampling methods are ready to use in one line. Convenience functions for interval selections, weighting regression ensembles, etc.

### Regression Modeling
Dozens of performance metrics, a few built in plots (Bland Altman, QQ, etc). CLS, Ridge, Kernel Ridge, LS-SVM, PCR, PLS(1/2), ELM's, ... More to come. Chemometricians love regressions!

### Classification Analysis
Inhouse classification encodings(one cold/one hot), multiclass performance statistics. Package includes: LDA with guassian discriminants, logistic regression, PLS-DA, K-NN, and more to come.  

## Specialized tools?
You might be saying, ridge regression, logistic regression, K-NN, PCA, etc, isn't this just a machine learning library with some preprocessing tools for chemometrics? If that's what you wanna use it for be my guest. Seems kinda wrong and inefficient to PyCall scikit learn to do basic machine learning/analysis anyways... I'll be buffing up the machine learning methods available as time allows. But, no, this is slowly becoming chock-full of chemometric methods.

The package does have some specialized tools for chemometricians in special fields. For instance, fractional derivatives for the electrochemists (and the adventurous), Savitsky Golay smoothing, curve resolution for forensics, etc. There are certainly plans for a few other tools for analyzing chemical data that packages in other languages have seemingly left out. More to come. Stay tuned.

## ToDo:
  - Peak finding algorithms
  - BTEM / from scratch Simulated Annealing Optimizer, ...
  - Fast decision trees...
  - Time Series / soft-sensing stuff / Recursive regression methods
  - SIMCA, N-WAY PCA, and N-WAY PLS
  - Hyperspectral data preprocessing
  - ... Writing the Docs for all the code...
  - ... Writing hundreds of unit tests ...
  - ... ... Finding dozens of bugs ... ...
  - Maybe add Design of Experiment tools (Partial Factorial design, simplex, etc...)
  - Maybe add electrochemical simulations and optical simulations?
