# ChemometricsTools
This is an essential collection of tools to perform Chemometric analysis' in Julia. If you are uninformed as to what Chemometrics is, it could nonelegantly be stated as the marriage between datascience and chemistry. Traditionally it is a pile of linear algebra that is well reasoned by the physics and meaning of chemical measurements. This is some orthogonal to most fields of machine learning, but sometimes we too get desperate and break out pure machine learning methods.

The goals for this package are the following: rely only on basic dependencies for longevity and stability, essential algorithms should read similar to pseudocode in papers, and the tools provided should be fast, flexible, and reliable. That's the Julia way after-all. No code will directly call R, Python, C, etc. As such it will be written in Julia from the ground up.

## Ethos
Arrays Only: In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change.

Center-Scaling: None of the methods in this package will center and scale for you. This package won't waste your time by centering and scaling large chunks of data every-time you do a regression/classification.

Dependencies: Only base libraries (LinearAlgebra, StatsBase, Statistics, Plots) etc will be required. This is for longevity, and to provide a fast precompilation time. As wonderful as it is that other packages exist to do some of the internal operations this one needs, we won't have to worry about a breaking change made by an external author working out the kinks in a separate package. I want this to be long-term reliable even if I go MIA for a week or two.

### Package Status => Early View
This thing is brand new (~3 weeks old). Many of the tools available can already be used, and most of those are implemented correctly, but the documentation is slow coming. Betchya anything there are a few bugs hiding in the repo. So use at your own risk for now. In a week or so this should be functional and trustworthy, and at that point collaborators will be sought. This is an early preview for constructive criticism and spreading awareness.

### Transforms/Pipelines
Two design choices introduced in this package are "Transformations" and "Pipelines". We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer, etc.

Multiple transformations can be easily chained together and stored using "Pipelines". These are basically convenience functions, but are somewhat flexible. Pipelines allow for preprocessing and data transformations to be reused, chained, or automated for reliable analytic throughput.

# Model training
There are a few built-in's to make training models a snap. Philosophically I decided, that making wrapper functions to perform Cross Validation is not fair to the end-user. There are many cases where we want specialized CV's but we don't want to write nested for-loops that run for hours then debug them... Similarly, most people don't want to spend their time hacking into rigid GridSearch objects, or scouring stack exchange / package documentation. Especially when it'd be easier to write an equivalent approach that's self documenting from scratch. Instead, I used Julia's iterators to make K-Fold validations convenient, below is an example Partial Least Squares Regression CV.

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

That's great right? but, hey that was kind of slow. Knowing what we know about ALS based models, we can do the same operation in linear time with respect to latent factors by computing the most latent variables first and only recomputing the regression coefficients. An example of this is below,

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
There's also a bunch of tools for changes of basis such as: principal components analysis, linear discriminant analysis, orthogonal signal correction, etc. With those kinds of tools we can reduce the dimensions of our data and make classes more separable. So separable that trivial classification methods like a Gaussian discriminant can get us pretty good results. Below is an example analysis performed on mid-infrared spectra of strawberry purees and adulterated strawberry purees (yes fraudulent food items are a common concern).

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

Cool right? Well, we can now apply the same transformations to the test set and pull some multivariate Gaussians over the train set classes to see how we do identifying fraudulent puree's,

```julia
TestSet = Train_pca(snv(Test));
TestSet = lda(TestSet);
TestPreds = classifier(TestS; Factors  = 2);
MulticlassStats(TestPreds .- 1, TstLbl , Enc)
```
If you're following along you'll get ~92% F-measure. Not bad. You may also notice this package has a nice collection of performance metrics for classification, regression, and clustering. Anyways, I've gotten 100%'s with more advanced methods but this is a cute way to show off some of the tools currently available.

# Curve Resolution

So far NMF, SIMPLISMA, and MCR-ALS are included in this package. If you aren't familiar with them, they are used to extract spectral and concentration estimates from unknown mixtures in chemical signals. Below is an example of spectra which are composed of signals from a mixture of a 3 components.

![RAW](/images/curveres.png)

Now we can apply some base curve resolution methods,

![NMF](/images/NMF.png)
![SIMPLISMA](/images/SIMPLISMA.png)

and, apply MCR-ALS on say the SIMPLISMA estimates to further refine them (non-negativity constraints and normalization are available),

![MCRALS](/images/MCRALS.png)

Kind of like chromatography without waiting by a column all day. Neat right. Ironically MCR-ALS spectra look less representative of the actual pure spectral components known to be in the mixture. However, their concentration profiles derived from MCR-ALS are far superior to that of those from SIMPLISMA. You'll have to play with the code yourself to see.

## Clustering
Currently K-means and basic clustering metrics are on board. Hey if you want clustering methods check out Clustering.jl! They've done an awesome job.

## Time Series/Soft-Sensing
Right now Echo State Networks are on board. Lot's to do there!

## Specialized tools?
You might be saying, ridge regression, least squares, logistic regression, K-NN, PCA, etc, isn't this just a machine learning library with some preprocessing tools for chemometrics? If that's what you wanna use it for be my guest. Seems kinda wrong and inefficient to PyCall scikit learn to do basic machine learning/analysis anyways... I'll be buffing up the machine learning methods available as time allows.

But, the package does have some specialized tools for chemometricians in special fields. For instance, fractional derivatives for the electrochemists (and the adventurous), Savitsky Golay smoothing, curve resolution for forensics, bland-altman plots, etc. There are certainly plans for a few other tools for analyzing chemical data that packages in other languages have seemingly left out. More to come. Stay tuned.

## ToDo:
  - Peak finding algorithms
  - BTEM / from scratch Simulated Annealing Optimizer, ...
  - Fast decision trees...
  - Time Series / soft-sensing stuff / Recursive regression methods
  - Generic interval based ensemble model code.
  - SIMCA, N-WAY PCA, and N-WAY PLS
  - Hyperspectral data preprocessing
  - ... Writing the Docs for all the code...
  - ... Writing hundreds of unit tests ...
  - ... ... Finding dozens of bugs ... ...
  - Maybe add Design of Experiment tools (Partial Factorial design, simplex, etc...)
  - Maybe add electrochemical simulations and optical simulations?
