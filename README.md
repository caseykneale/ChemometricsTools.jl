No official doc's yet, but the [wiki](https://github.com/caseykneale/ChemometricsTools/wiki) has the majority of the API and a few tutorials. Feel free to check the code-base, good place to start would be the [module source](https://github.com/caseykneale/ChemometricsTools/blob/master/src/ChemometricsTools.jl).

# ChemometricsTools
This is a collection of tools to perform fundamental and advanced Chemometric analysis' in Julia. It is currently richer then any single free chemometrics package available in any other language. If you are uninformed as to what Chemometrics is; it could nonelegantly be described as the marriage between data science and chemistry. Traditionally it is a pile of applied linear algebra that is well reasoned by the physics and meaning of chemical measurements. This is somewhat orthogonal to most fields of machine learning(aka "add more layers"), but sometimes chemometricians also get desperate and break out pure machine learning methods. So some of those are in here, but if you want neural networks [Flux.jl](https://github.com/FluxML/Flux.jl) is my favorite deep learning library; it is super easy and fast to use.

The goals for this package are the following: rely only on basic dependencies for longevity and stability, essential algorithms should read similar to pseudocode in papers, and the tools provided should be fast, flexible, and reliable. That's the Julia way after-all. No code will directly call R, Python, C, etc. It is currently and will continue to be written in Julia from the ground up.

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
Dependencies: Only base libraries (LinearAlgebra, StatsBase, Statistics, Plots) etc will be required. This is for longevity, and to provide a fast precompilation time. As wonderful as it is that other packages exist to do some of the internal operations this one needs, we won't have to worry about a breaking change made by an external author working out the kinks in a separate package. I want this to be long-term reliable without much upkeep. I'm a busy guy working a day job; I write this to warm-up before work, and unwind afterwards.

Arrays Only: In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change.

Center-Scaling: None of the methods in this package will center and scale for you. This package won't waste your time deciding if it should auto-center/scale large chunks of data every-time you do a regression/classification.

### Package Status => Early View
This thing is brand new (< 3 weeks old). Many of the tools available can reliably be used, but the documentation definitely lags behind progress right now. Betchya anything there are a few bugs hiding in the repo. In a week or so this should be functional and trustworthy, and at that point collaborators will be sought. This is an early preview for constructive criticism and spreading awareness.

# Package Highlights
### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer, etc.

Multiple transformations can easily be chained together and stored using "Pipelines". Pipelines aren't "pipes" like are present in Bash, R and base Julia. They are flexible, yet immutable, convenience objects that allow for sequential preprocessing and data transformations to be reused, chained, or automated for reliable analytic throughput.

### Model training
Easy to use iterators for K-Folds validation's! Sampling methods like Kennard Stone, and resampling methods are each one function call away from use. Convenience functions for interval selections, weighting regression ensembles, etc are also available. This allows for building models like SIPLS, P-DS, P-OSC, etc to be made quickly.

### Regression Modeling
Dozens of regression performance metrics, and a few built in plots (Bland Altman, QQ, Interval Overlays etc) are included. ChemometricsTools currently includes: CLS, Ridge, Kernel Ridge, LS-SVM, PCR, PLS(1/2), ELM's, ... More to come. Chemometricians love regressions!

### Classification Analysis
In-house classification encodings(one cold/one hot), multiclass performance statistics. ChemometricsTools currently includes: LDA with guassian discriminants, logistic regression, PLS-DA, K-NN, naive bayes, classification trees, and more to come.

## Specialized tools?
You might be saying, classification trees, ridge regression, logistic regression, K-NN, PCA, etc, isn't this just a machine learning library with some preprocessing tools for chemometrics? If that's what you wanna use it for be my guest. Seems kinda wrong and inefficient to PyCall scikit learn (which calls on C, fortran, etc) to do basic machine learning/analysis in Julia anyways... I'll be buffing up the machine learning methods available as time allows. But, no, this is slowly becoming chock-full of pure chemometric methods.

The package does have specialized tools for chemometricians in special fields. For instance, fractional derivatives for the electrochemists (and the adventurous), Savitsky Golay smoothing, curve resolution for forensics, etc. There are certainly plans for a few other tools for analyzing chemical data that packages in other languages have seemingly left out. More to come. Stay tuned.

## Why Julia?
In Julia we can do mathematics like R or Matlab (no installations/imports), but write glue code as easily as python, with the expressiveness of scala, with (often) the performance of C/C++. Multidispatch makes recycling code painless, and broadcasting allows for intuitive application of operations across collections. I'm not a soft-ware engineer, but, these things have made Julia my language of choice.

## Wheres the Data?
Right now I don't have rights to any data. I'd love for a collaborator to contribute some high-quality: spectra, mass spectrums, chromatograms, etc. Please reach out to me if you wish to collaborate with proper credit. There's a good chance in a week or so I'll be reaching out to the community for these sorts of things.

## ToDo:
  - Docs (Update): Peakfinding, NB, tree methods, MaxVoteOneHot, boxcar, boxcox, norms, snv, RandomForest, PSO...
  - Docs (Make): Method shoot-out for regression/classification/etc...
  - BTEM...
  - Time Series / soft-sensing stuff / Recursive regression methods
  - SIMCA, N-WAY PCA, and N-WAY PLS
  - ... Writing hundreds of unit tests ...

## Maybes:
  - Hyperspectral data preprocessing?
  - Convenience fns for standard addition, propagation of error, multiequilibria, kinetics?
  - Design of Experiment tools (Partial Factorial design, simplex, etc...)
  - Electrochemical simulations and optical simulations?
