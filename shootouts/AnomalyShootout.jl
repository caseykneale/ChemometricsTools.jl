using Pkg
Pkg.activate(".")
using ChemometricsTools
using Plots
#View the data in the package space
ChemometricsToolsDatasets()
#Load the data from package-space: remove the time column while we're at it
DF = ChemometricsToolsDataset("actuator.csv");
actuator = convert(Matrix,  DF)[:,2:end];
#Split into a training set based on normal behaviour
(Train, Test) = SplitByProportion(actuator, 0.40)
#Plot the data
plot(Train, legend = false, title = "Train Set - Raw Data", ylabel = "Measurements", xlabel = "time")
#alot of DC offsets let's column center each variable.
clean = Center(Train);
pca = PCA(clean(Train); Factors = 5);
cumsum( ExplainedVariance( pca ) )
(Lambdas, limits) = Hotelling(clean(Train), pca; Quantile = 0.015, Variance = 0.95)
scatter(Lambdas, title = "Train - T2 vs Time", xlabel = "time", ylabel = "T2", label = "T2");
hline!([limits], label = "Control Limit")

(qs, limits) = Q(clean(Train), pca; Quantile = 1.0 - 1e-16, Variance = 0.99)
scatter(qs, title = "Train - Q vs Time", xlabel = "time", ylabel = "Q", label = "Q");
hline!([limits], label = "Control Limit")


#Let's apply them to the test set...
(Lambdas, limits) = Hotelling(clean(Test), pca; Quantile = 0.015, Variance = 0.95);
scatter(Lambdas, title = "Test - T2 vs Time", xlabel = "time", ylabel = "T2", label = "T2");
hline!([limits], label = "Control Limit")

(qs, limits) = Q(clean(Test), pca; Quantile = 1.0 - 1e-16, Variance = 0.99);
scatter(qs, title = "Test - Q vs Time", xlabel = "time", ylabel = "Q", label = "Q");
hline!([limits], label = "Control Limit")
vline!([120], label = "Known Fault")
