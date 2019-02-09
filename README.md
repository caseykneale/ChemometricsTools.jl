# ChemometricsTools
This is an essential collection of tools to perform Chemometric analysis' in Julia. The goals for this package are as follows: rely only on basic dependencies for longevity and stability, essential algorithms should read similar to pseudocode in papers, and the tools provided should be fast, flexible, and reliable (That's the Julia way after-all). No code will directly call R, Python, C, etc, it will be written in Julia from the ground up.

In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change. For now the package is best suited to the treatment and analysis of continuous data. Surely there's plans to broaden the scope.

### Package Status => Early View
This thing is brand new (~2 weeks old). Many of the tools available can be used, and most of those are implemented correctly. Betchya anything there are bugs in the repo! So use at your own risk for now. In a week or two this should be functional and trustworthy, and at that point collaborators will be sought. I'm releasing an early preview for constructive criticism and awareness.

### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". These allow for preprocessing and data transformations to be reused or chained for reliable analytic throughput. Below are some examples based on some faux data,
```julia
FauxSpectra1 = randn(10,200);
FauxSpectra2 = randn(5, 200);
```
#### Transformations
We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer, etc.

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
We can take advantage of how pipelines are created; at their core they are tuples of transforms/functions. So if we can make an array of transforms and set some conditions they can be stored and applied to unseen data. A fun example of an automated transform pipeline is in the whimsical paper written by Willem Windig et. al's. That paper is called 'Loopy Multiplicative Scatter Transform'. Below I'll show how we can implement that algorithm here (or anything similar) with ease.
*Loopy MSC: A Simple Way to Improve Multiplicative Scatter Correction. Willem Windig, Jeremy Shaver, Rasmus Bro. Applied Spectroscopy. 2008. Vol 62, issue: 10, 1153-1159*

First let's look at the classic Diesel data before applying Loopy MSC
![Before Loopy MSC](/images/Raw.png)

Alright, there is scatter, let's go for it,
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
#Now we can make a pipeline object from the array of stored transforms
LoopyPipe = Pipeline( Tuple( TransformArray ) );
```
For a sanity check we can ensure the output of the algorithm  is the same as the new pipeline so it can be applied to new data.
```julia
Current == LoopyPipe(RealSpectra)
```
Looks like our automation driven pipeline is equivalent to the loop it took to make it. More importantly did we remove scatter after 3 automated iterations of MSC?

![After Loopy MSC](/images/Loopy.png)

Yes, yes we did. Pretty easy right?

# Model training
There are a few built-in's to make training models a snap. Philosophically I decided, making wrapper functions to perform Cross Validation is not fair to the end-user. There are many cases where we want specialized CV's but we don't want to write nested for-loops that run for hours then debug them... Similarly, most people don't want to spend their time hacking into rigid GridSearch code, scouring stack exchange and package documentation. Especially when it'd be easier to write an equivalent approach that's self documenting from scratch. Instead, I used Julia's iterators to make K-Fold validations convenient, below is an example Partial Least Squares Regression CV.

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

That's great right? but, hey that was kinda slow. Knowing what we know about ALS based models, we can do the same operation in linear time with respect to factors by computing the most latent variables first and only recomputing the regression coefficients. An example of this is below,

```julia
Err = repeat([0.0], 22);
Models = []
for Lv in 22:-1:1
    for ( i, ( Fold, HoldOut ) ) in enumerate(KFoldsValidation(20, TrainX, TrainY))
        if Lv == 22
            push!( Models, PartialLeastSquares(Fold[1], Fold[2]; Factors = Lv) )
        end
        Err[Lv] += SSE( Models[i]( HoldOut[1]; Factors = Lv), HoldOut[2] )
    end
end
```
This approach is ~5 times faster on a single core( < 2 seconds), pours through 7Gb less data, and makes 1/5th the allocations. If you wanted you could distribute the inner loop (using Distributed.jl) and see drastic speed ups!

*Aside:* there are quite a few other functions that make model training convenient for end-users. Such as Shuffle, Shuffle!, LeaveOneOut, Venetian Blinds, etc.

The lovely Kennard-Stone sampling algorithm is also on board,
![Kennard-Stone](/images/KS.png)

# Classification Analysis
There's also a bunch of tools for changes of basis such as: principal components analysis, linear discriminant analysis, orthogonal signal correction, etc. With those kinda of tools we can reduce the dimensions of our data and make classes more seperable. So seperable that trivial classification methods like gaussian discriminant can get us pretty good results. Below is an example analysis performed on mid-infrared spectra of strawberry purees and adulterated strawberry purees (yes fraudulent food items are a common concern).

![Raw](/images/fraud_analysis_raw.png)

*Use of Fourier transform infrared spectroscopy and partial least squares regression for the detection of adulteration of strawberry purées. J K Holland, E K Kemsley, R H Wilson*


```julia
snv = StandardNormalVariate(Train);
Train_pca = PCA(snv(Train);; Factors = 15);

Enc = LabelEncoding(TrnLbl);
Hot = ColdToHot(TrnLbl, Enc);

lda = LDA(Train_pca.Scores , Hot);
classifier = GaussianDiscriminant(lda, TrainS, Hot)
TrainPreds = classifier(TrainS; Factors = 2);
```
![LDA of PCA](/images/lda_fraud_analysis.png)

Cool right? Well, we can now apply the same transformations to the test set and pull some multivariate guassians over the train set classes to see how we do identifying fraudulent puree's,

```julia
TestSet = Train_pca(snv(Test));
TestSet = lda(TestSet);
TestPreds = classifier(TestS; Factors  = 2);
MulticlassStats(TestPreds .- 1, TstLbl , Enc)
```
If you're following along you'll get ~92% F-measure. Not bad. I've gotten 100%'s with more advanced methods but this is a cute way to show off some of the tools currently available.

# Clustering
Currently K-means and basic clustering metrics are on board. Hey if you want clustering methods check out Clustering.jl! They've done an awesome job.

#Time Series
Write now we have echo state networks on board. Lot's to do there!

# Specialized tools?
You might be saying, ridge regression, least squares, KNN, PCA, etc, isn't this just a machine learning library with some preprocessing tools for chemometrics? Right now, that's kind of true.

Well, we have some specialized tools for chemometricians in special fields. For instance, fractional derivatives for the electrochemists (and the adventurous), Savitsky Golay smoothing, and there are certainly plans for a few other tools for chemical data that packages in other languages have left out. Stay tuned... Right now some bare bones stuff still needs to be tuned for correctness, and some analysis functions need to be added. This is again, an early view!


# ToDo:
  - Finish PCA statistics
  - Convenience functions for Plots/Bland Altmans
  - Peak finding algorithms
  - Logit Transforms
  - Logistic/Multimodal regression
  - SIMCA
  - Time Series/soft-sensing stuff / Recursive regression methods
  - N-WAY PCA, and PLS
  - ... Writing hundreds of unit tests ...
  - ... ... Finding dozens of bugs ... ...
