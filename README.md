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

### Automation Example
We can take advantage of how pipelines are created; at their core they are tuples of transforms/functions. So if we can make an array of transforms and set some conditions we can store them and apply them. A fun example of this is the whimsical paper by Willem Windig et. al's for 'Loopy Multiplicative Scatter Transform', which we can implement here with ease.
*Loopy MSC: A Simple Way to Improve Multiplicative Scatter Correction. Willem Windig, Jeremy Shaver, Rasmus Bro. Applied Spectroscopy. 2008. Vol 62, issue: 10, 1153-1159*

Let's look at the classic Diesel data before loopy MSC
![Before Loopy MSC](/images/Raw.png)

```julia
RealSpectra = convert(Array, CSV.read("/diesel_spectra.csv"));
Current = RealSpectra;
Last = zeros(size(Current));
TransformArray = [];
while RMSE(Last, Current) > 1e-5
    if any(isnan.(Current))
        break
    else
        push!(TransformArray, MultiplicativeScatterCorrection( Current ) )
        Last = Current
        Current = TransformArray[end](Last)
    end
end
#Make the pipeline object from the array of stored transforms
LoopyPipe = Pipeline( Tuple( TransformArray ) );
```
For a sanity check we can ensure the output of the algorithm  is the same as the new pipeline so it can be applied to new data.
```julia
Current == LoopyPipe(RealSpectra)
```
More importantly did we remove scatter after 3 automated iterations of MSC?
![After Loopy MSC](/images/Loopy.png)

Yes, yes we did. Easy right?

# Model training
There are a few built-in's to make training models a snap. Philosophically I decided, making wrapper functions to perform Cross Validation is not fair to the end-user. There are many cases where we want specialized CV's and we don't want to write then debug nested for-loops that run for hours. Or hacking into rigid GridSearch code when it'd be easier to write the equivalent from scratch. Instead, I used Julia's iterators to make K Fold's validations convenient, below is an example Partial Least Squares Regression CV.

```julia
#Split our data
((TrainX,TrainY),(TestX, TestY)) = SplitByProportion(x, yprop, 0.7);
#Preprocess it
MSC_Obj = MultiplicativeScatterCorrection(TrainX);
TrainX = MSC_Obj(TrainX);
TestX = MSC_Obj(TestX);
#Begin CV!
LatentVariables = 22
Err = repeat([0.0], LatentVariables);
#Note this is the Julian way to nest two loops
for Lv in 1:LatentVariables, (Fold, HoldOut) in KFoldsValidation(20, TrainX, TrainY)
    PLSR = PartialLeastSquares(Fold[1], Fold[2]; Factors = Lv)
    Err[Lv] += SSE( PLSR(HoldOut[1]), HoldOut[2] )
end
scatter(Err, xlabel = "Latent Variables", ylabel = "Cumulative SSE", labels = ["Error"])
BestLV = argmin(Err)
PLSR = PartialLeastSquares(TrainX, TrainY; Factors = BestLV)
RMSE( PLSR(TestX), TestY )
```
![20 fold cross validation](/images/CV.png)
*Note:* there are quite a few other functions that make model training convenient for end-users. Such as Shuffle, Shuffle!, LeaveOneOut, Venetian Blinds, Kennard Stone, etc.
