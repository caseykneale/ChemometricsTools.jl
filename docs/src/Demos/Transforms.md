### Transforms Demo
Two design choices introduced in this package are "Transformations" and "[Pipelines](https://github.com/caseykneale/ChemometricsTools/wiki/Pipelines)". Transformations are the smallest unit of a 'pipeline'. They are simply functions that have a deterministic inverse. For example if we mean center our data and store the mean vector, we can always invert the transform by adding the mean back to the data. That's effectively what transforms do, they provide to and from common data transformations used in chemometrics.

Let's start with a trivial example with faux data where a random matrix of data is center scaled and divided by the standard deviation:
```julia
FauxSpectra1 = randn(10,200);
CS = CenterScale(FauxSpectra1);
Transformed1 = CS(FauxSpectra1);
```
As can be seen the application of the CenterScale() function returns an object that is used to transform future data by the data it was created from. This object can be applied to new data as follows,
```julia
FauxSpectra2 = randn(10,200);
Transformed2 = CS(FauxSpectra2);
```
Transformations can also be inverted (with-in numerical noise). For example,
```julia
RMSE(FauxSpectra1, CS(Transformed1; inverse = true)) < 1e-14
RMSE(FauxSpectra2, CS(Transformed2; inverse = true)) < 1e-14
```

We can use transformations to treat data from multiple sources the same way. This helps mitigate user-error for cases where test data is scaled based on training data, calibration transfer, etc. [Pipelines](https://github.com/caseykneale/ChemometricsTools/wiki/Pipelines) are a logical and convenient extension of transformations.
