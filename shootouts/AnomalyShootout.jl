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


#Let's apply the control limits obtained from the train set to the test set...
(Lambdas, limits) = Hotelling(clean(Test), pca; Quantile = 0.015, Variance = 0.95);
scatter(Lambdas, title = "Test - T2 vs Time", xlabel = "time", ylabel = "T2", label = "T2");
hline!([limits], label = "Control Limit");
vline!([120], label = "Known Fault")

(qs, limits) = Q(clean(Test), pca; Quantile = 1.0 - 1e-16, Variance = 0.99);
scatter(qs, title = "Test - Q vs Time", xlabel = "time", ylabel = "Q", label = "Q");
hline!([limits], label = "Control Limit");
vline!([120], label = "Known Fault")

#Hotellings T2 is slower to detect the failure then the Q plot. Not a surprise but good to see.

#Let's try an EWMA on a single variable...
plot(Train[:,3], title = "Variable #1", xlabel = "time", ylabel = "Magnitude")
#Use the "Burn in" constructor for EWMA
EWM = EWMA(Train[:,3], 0.1)
(UpperEWMA, LowerEWMA) = Limits(EWM; k = 3)

#Let's get to predicting already...
ewmascore = []
for s in 1:length( Test[:,3] )
    push!( ewmascore, EWM( Test[s,3] ; train = false) )
end
plot(ewmascore, label = "EWMA", title = "EWMA Fault detection", ylabel = "EWMA", xlabel = "time");
hline!([UpperEWMA], label = "Upper Limit");
hline!([LowerEWMA], label = "Lower Limit");
vline!([120], label = "Known Fault")

score = OneClassJKNN( Train, Test; J = 1, K = 1,  DistanceType = "euclidean" );
scatter(log10.(score));
hline!([1.0], label = "Threshold of 1");
vline!([120], label = "Known Fault")
