using Pkg
Pkg.activate(".")
using ChemometricsTools
ChemometricsToolsDatasets()
#Load the data from package-space
Tecator = convert(Matrix, ChemometricsToolsDataset("tecator.csv") )
#Seperate data from properties
