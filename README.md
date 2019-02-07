# ChemometricsTools
This is an essential collection of tools to do Chemometrics in Julia. The goals for this package are as follows: only rely on basic dependencies, essential algorithms should read similar to pseudocode in papers, and provide flexible tooling for the end-user's fast and reliable work flow.

In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change. For now the package is best suited to the treatment and analysis of continuous data.

### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". These allow for preprocessing and data transformations to be reused or chained for reliable analytic throughput. Below are some examples based on some faux data,
```julia
FauxSpectra1 = randn(10,200);
FauxSpectra2 = randn(5, 200);
```
#### Transformations
We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer etc.

```julia
SNV = StandardNormalVariate(FauxSpectra1);
Transformed1 = SNV(FauxSpectra1);
Transformed2 = SNV(FauxSpectra2);
```
Transformations can also be inverted(within numerical noise). For example,
```julia
RMSE(FauxSpectra1, SNV(Transformed1; inverse = true)) < 1e-14
RMSE(FauxSpectra2, SNV(Transformed2; inverse = true)) < 1e-14
```
#### Pipelines
Multiple transformations can be easily chained together and stored using "Pipelines". These are basically convenience functions, but are somewhat flexible and can be used for automated searches,
```julia
PreprocessPipe = Pipeline(FauxSpectra1, RangeNorm, Center);
Processed = PreprocessPipe(FauxSpectra1);
```
Of course pipelines of transforms can also be inverted,
```julia
RMSE( FauxSpectra1, PreprocessPipe(Processed; inverse = true) ) < 1e-14
```
Pipelines can also be created and executed as an 'in place' operation for large datasets. This has the advantage that your data is transformed immediately without making copies in memory. This may be useful for large datasets and memory constrained environments.
*WARNING:* be careful to only run the pipeline call or its inverse once! It is much safer to use the not inplace function outside of a REPL/script environment.

```julia
FauxSpectra = randn(10,200);
OriginalCopy = copy(FauxSpectra);
InPlacePipe = PipelineInPlace(FauxSpectra, Center, Scale);
```
See without returning the data or an extra function call we have transformed it according to the pipeline as it was instantiated...
```julia
FauxSpectra == OriginalCopy
#Inplace transform the data back
InPlacePipe(FauxSpectra; inverse = true)
RMSE( OriginalCopy, FauxSpectra ) < 1e-14
```
Pipelines are kind of flexible. We can put nontransform (operations that cannot be inverted) preprocessing steps in them as well. In the example below the first derivative is applied to the data, this irreversibly removes a column from the data,
```julia
PreprocessPipe = Pipeline(FauxSpectra1, FirstDerivative, RangeNorm, Center);
Processed = PreprocessPipe(FauxSpectra1);
#This should be equivalent to the following...
SpectraDeriv = FirstDerivative(FauxSpectra1);
Alternative = Pipeline(SpectraDeriv , RangeNorm, Center);
Processed == Alternative(SpectraDeriv)
```
Great right? Well what happens if we try to do the inverse of our pipeline with an irreversible function (First Derivative) in it?
```julia
PreprocessPipe(Processed; inverse = true)
```

Well we get an assertion error.
