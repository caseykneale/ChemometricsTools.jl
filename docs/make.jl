# push!(LOAD_PATH, "/home/caseykneale/Desktop/ChemometricsTools/ChemometricsTools.jl/");
# using Pkg
# Pkg.activate(".")
using Documenter, ChemometricsTools

makedocs(
	sitename = "ChemometricsTools",
	modules = [ChemometricsTools],
	format = Documenter.HTML(# Use clean URLs, unless built as a "local" build
	        prettyurls = !("local" in ARGS),
		),
	authors = "Casey Kneale",
    doctest = true,
	pages = ["Home" => "index.md",
			"Demos" => Any[
						"Transforms" => "Demos/Transforms.md",
						"Pipelines" => "Demos/Pipelines.md",
						"Classification" => "Demos/ClassificationExample.md",
						"Regression" => "Demos/RegressionExample.md",
						"SIPLS" => "Demos/SIPLS.md",
						"Calibration Transfer" => "Demos/CalibXfer.md",
						"Curve Resolution" => "Demos/CurveResolution.md"
					   ],
			 "Data" => Any[
			 					"Data Utilities" => "man/datautils.md",
			 			 		"Kernel Density Generator" => "man/kerneldensity.md"
							],
			 "Manipulations" => Any[
			 					"Preprocessing" => "man/Preprocess.md",
			 			 		"Transformations/Pipelines" => "man/Transformations.md",
						 		"Sampling" => "man/Sampling.md"
							],
			 "Visualizations" => Any[ "Plotting" => "man/Plotting.md"],
			 "Analysis" => Any[
			 					"DOE" => "man/DOE.md",
								"Univariate" => "man/Univariate.md",
			 					"Distances/Kernels" => "man/Dists.md",
			 					"Analysis" => "man/Analysis.md",
			  					"Clustering" => "man/Clustering.md",
								"Stats." => "man/Stats.md"
							],
			 "Modeling" => Any[
			 					"Training" => "man/Training.md",
						 		"Time Series" => "man/TimeSeries.md",
						 		"Regression Models" => "man/RegressionModels.md",
						 		"Regression Metrics" => "man/regressMetrics.md",
			 			 		"Classification Models" => "man/ClassificationModels.md",
						 		"Classification Metrics" => "man/classMetrics.md",
						 		"Tree Methods" => "man/Trees.md",
						 		"Ensemble Models" => "man/Ensemble.md"
							],
			 "Speciality Tools" => Any[
			 					 "Model Analysis" => "man/modelanaly.md",
						 		 "MultiWay" => "man/MultiWay.md",
								 "Hyperspectral" => "man/Hyperspectral.md",
								 "Anomaly Detection" => "man/AnomalyDetection.md",
								 "Curve Resolution" => "man/CurveResolution.md"
							],
			"Advanced" => Any[
								 "PSO" => "man/PSO.md",
								 "Genetic Algorithms" => "man/GeneticAlgorithms.md"
		    			]
		]
)

deploydocs(
    repo = "github.com/caseykneale/ChemometricsTools.jl.git",
	branch = "gh-pages",
)
