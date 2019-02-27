using Pkg
Pkg.activate(".")
using ChemometricsTools
#View the data in the package space
ChemometricsToolsDatasets()
#Load the data from package-space
DF = ChemometricsToolsDataset("tecator.csv");

Xindices = findall(occursin.( "absorbance", string.(names(DF) ) ) );
names(DF)[(end - 2) : end]
Moisture = DF[ 123 ];
Tecator = convert(Matrix,  DF[:,Xindices]);
Df = 0; #Clear the memory...


using Plots
plot(Tecator', legend = false, ylabel = "Absorbance", xlabel = "bins")


msc = MultiplicativeScatterCorrection(Tecator)
#Shuffle the set, because the classes are ordered
(Tecator, Moisture) = Shuffle(Tecator, Moisture)
#Split the data into a training and hold out set
((TrainX, TrainY), (TestX, TestY)) = SplitByProportion(Tecator, Moisture, 0.7)
#Can't have spectra and not plot it!
plot(msc(TrainX)', legend = false, ylabel = "Absorbance", xlabel = "bins")

#Let the shoot out begin!
