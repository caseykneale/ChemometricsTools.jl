push!(LOAD_PATH,"/src/")
using Pkg
Pkg.activate(".")
using Documenter, ChemometricsTools

makedocs(
	sitename = "ChemometricsTools",
	modules = [ChemometricsTools],
	format = Documenter.HTML(),
    	authors = "Casey Kneale",
    	doctest = true
)
