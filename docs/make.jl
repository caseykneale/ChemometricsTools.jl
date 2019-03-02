push!(LOAD_PATH,"/src/")
using Pkg
Pkg.activate(".")
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
			 "API" => Any[
			 			 "Preprocessing" => "man/Preprocess.md",
			 			 "Transformations/Pipelines" => "man/Transformations.md",
						 "Sampling" => "man/Sampling.md",
						 "Training" => "man/Training.md",
						 "Time Series" => "man/TimeSeries.md",
						 "Regression Models" => "man/RegressionModels.md",
						 "Regression Metrics" => "man/RegressionMetrics.md",
			 			 "Classification Models" => "man/ClassificationModels.md",
						 "Classification Metrics" => "man/ClassificationMetrics.md",
						 "Tree Methods" => "man/Trees.md",
						 "Ensemble Models" => "man/Ensemble.md",
						 "Clustering" => "man/Clustering.md",
						 "Anomaly Detection" => "man/AnomalyDetection.md",
						 "Curve Resolution" => "man/CurveResolution.md",
						 "Stats." => "man/Stats.md",
						 "Distance Measures" => "man/Dists.md",
						 "PSO" => "man/PSO.md",
		    			],
			 "Full API" => "man/FullAPI.md",
		]
)
