#using Pkg
#pwd()
#Pkg.activate(".")
using ChemometricsTools
#View the data in the package space
ChemometricsToolsDatasets()
#Load the data from package-space
DF = ChemometricsToolsDataset("tecator.csv");

Xindices = findall(occursin.( "absorbance", string.(names(DF) ) ) );
names(DF)[(end - 2) : end]
Moisture = DF[ 123 ];
Tecator = convert(Matrix,  DF[:,Xindices]);
#Let's check the Tecator data for duplicate entries ;)
length(AssessHealth(Tecator)["Duplicates"])
#Eeek plenty of duplicates, oh well this is a shoot out not a real analysis...
Df = 0; #Clear the memory...


using Plots
#plot(Tecator', legend = false, ylabel = "Absorbance", xlabel = "bins")


#Shuffle the set, because the classes are ordered
(Tecator, Moisture) = Shuffle(Tecator, Moisture);
#Split the data into a training and hold out set
((TrainX, TrainY), (TestX, TestY)) = SplitByProportion(Tecator, Moisture, 0.7);
msc = MultiplicativeScatterCorrection(TrainX);
TrainX = msc(TrainX);
TestX = msc(TestX);

#Can't have spectra and not plot it!
#plot(TrainX', legend = false, ylabel = "Absorbance", xlabel = "bins")


#Let the shoot out begin!
@time cls   = ClassicLeastSquares( TrainX[:, [5, 25, 45, 60, 90] ], TrainY; Bias = true )
@time rr    = RidgeRegression( TrainX, TrainY, 0.1; Bias = true )
@time krr   = KernelRidgeRegression( TrainX, TrainY, 0.01; KernelParameter = 1.0, KernelType = "gaussian" )
@time lssvm = LSSVM( TrainX, TrainY, 0.01; KernelParameter = 1.0, KernelType = "gaussian" )
pca = PCA(TrainX;Factors = 4);
@time pcr   = PrincipalComponentRegression(pca, TrainY )
@time plsr  = PartialLeastSquares( TrainX, TrainY; Factors = 4, tolerance = 1e-8, maxiters = 200 )
@time elm   = ExtremeLearningMachine(TrainX, TrainY, 11; ActivationFn = sigmoid)
@time rt    = RegressionTree( TrainX, TrainY; gainfn = ssd, maxdepth = 15, minbranchsize = 5)
@time rf    = RandomForest(TrainX, TrainY, :regression; gainfn = ssd, trees = 100,
                       maxdepth = 15,  minbranchsize = 5,
                       samples = 0.7, maxvars = nothing)

#Let's evaluate them on the training set then the hold out - see how they do!
models = ( rr, krr, lssvm, pcr, plsr, elm, rt, rf );
modelnames = ( :rr, :krr, :lssvm, :pcr, :plsr, :elm, :rt, :rf );

TrainingRMSE = Dict()
TrainingRMSE[:cls] = RMSE( cls(TrainX[:,[5, 25, 45, 60, 90]]), TrainY )
for ( name, model ) in zip(modelnames, models)
    TrainingRMSE[name] = RMSE( model(TrainX), TrainY )
    if name == :rf
        println(model(TrainX))
    end
end

TrainingRMSE

HoldOutRMSE = Dict()
HoldOutRMSE[:cls] = RMSE( cls(TestX[:,[5, 25, 45, 60, 90]]), TestY )
for ( name, model ) in zip(modelnames, models)
    HoldOutRMSE[name] = RMSE( model(TestX), TestY )
end
HoldOutRMSE
