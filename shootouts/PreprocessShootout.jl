using Pkg
pwd()
cd("../ChemometricsTools.jl/")
Pkg.activate(".")

using ChemometricsTools

#View the data in the package space
ChemometricsToolsDatasets()
#Load the data from package-space
DF = ChemometricsToolsDataset("tecator.csv");

Xindices = findall(occursin.( "absorbance", string.(names(DF) ) ) );
Tecator = convert(Matrix,  DF[:,Xindices]);

#Let's check the Tecator data for duplicate entries ;)
length(AssessHealth(Tecator)["Duplicates"])
#Eeek plenty of duplicates, oh well this is a shoot out not a real analysis...

using Plots
plot(Tecator', legend = false, ylabel = "Absorbance", xlabel = "bins")
using Statistics
BC = boxcar(Tecator[1,:]; windowsize = 5, fn = mean);
AS = ALSSmoother(Tecator[1,:]; lambda = 100, p = 0.1, maxiters = 10);
PS = PerfectSmoother(Tecator[1,:]; lambda = 100);
SG = SavitzkyGolay(convert.(Float64, Tecator[1,:])', 0, 1, 3);

plot( [ Tecator[1,:], BC', AS', PS', SG' ], xlabel = "bins", ylabel = "Smoothed Signal",
        label = [ "Original", "BC", "AS", "PS", "SG" ], ylim = (2.5,3.75) )

FD = FirstDerivative(Tecator[1,:]);
SD = SecondDerivative(Tecator[1,:]);
HD = FractionalDerivative(convert.(Float64, Tecator[1,:]); Order = 1.5);

plot( [ FD', SD', HD' ], xlabel = "bins", ylabel = "Differentiated Signal",
        label = [ "First Derivative", "Second Derivative", "1(1/2) Derivative" ],
        ylim = (-0.1,0.1) )
