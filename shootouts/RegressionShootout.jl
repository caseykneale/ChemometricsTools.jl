using Pkg
Pkg.activate(".")
using ChemometricsTools
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

(Tecator, Moisture) = Shuffle(Tecator, Moisture)

((TrainX, TrainY), (TestX, TestY)) = SplitByProportion(Tecator, Moisture, 0.7)
plot(msc(TrainX)', legend = false, ylabel = "Absorbance", xlabel = "bins")
