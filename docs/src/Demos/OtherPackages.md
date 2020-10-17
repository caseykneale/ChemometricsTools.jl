# Other Packages:
So you know what you're doing, you're not one of those Friday night chemometricians Brereton talked about, and you want to compare some methods available in ChemometricsTools.jl. Great! The nice thing about Julia is, packages tend to work with one another with zero effort. To demonstrate this I made a little tutorial using (Turing.jl)[https://turing.ml/dev/] and (ChemometricsData.jl)[https://github.com/caseykneale/ChemometricsData.jl] for a very basic incomplete analysis of some well known bayesian regression methods. Let's get started.


## Lets load in some data
```julia
using Turing, StatsPlots, Plots, Statistics
using DataFrames, ChemometricsData, ChemometricsTools

println( ChemometricsData.search("corn") )
corn_data = ChemometricsData.load("Cargill_Corn")
X = Matrix(corn_data["m5_spectra.csv"])

xaxis = 1100:2:2498#nm

plot( X', title = "Cargill Corn M5 Spec", xlab = "Wavelength (nm)", ylab = "Absorbance", legend = false,
        xticks = (1:50:length(xaxis), xaxis[1:50:end]) )
```
![CornEDA](https://raw.githubusercontent.com/caseykneale/ChemometricsTools/master/images/BayesDemo/CornSpectra.png)

Grab our property values,
```julia
Y = corn_data["property_values.csv"][!,:Moisture]
```

Now let's center and scale our X and Y values to keep our regression methods happy

```julia
train, test = 1:35, 36:80
X_train, X_test = X_processed[train,:], X_processed[test,:]
μx,σx = mean(X_train, dims = 1), std(X_train, dims = 1)
X_train = (X_train .- μx) ./ σx
X_test = (X_test .- μx) ./ σx

Y_train, Y_test = Y[train,:], Y[test,:]
μy,σy = mean(Y_train),std(Y_train)
Y_train = (Y_train .- μy) ./ σy
Y_test = (Y_test .- μy) ./ σy
```


![20folds](https://raw.githubusercontent.com/caseykneale/ChemometricsTools/master/images/CV.png)

That's great right? but, hey that was kind of slow. Knowing what we know about ALS based models, we can do the same operation in linear time with respect to latent factors by computing the most latent variables first and only recomputing the regression coefficients. An example of this is below,

```julia

```
This approach is ~5 times faster on a single core( < 2 seconds), pours through 7Gb less data, and makes 1/5th the allocations (on this dataset at least). If you wanted you could distribute the inner loop (using Distributed.jl) and see drastic speed ups!
