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
						"Curve Resolution" => "Demos/CurveResolution.md",
					   ],
			 "API" => Any[
			 			 "Kernel Density Generator" => "man/kerneldensity.md",
			 			 "Preprocessing" => "man/Preprocess.md",
			 			 "Transformations/Pipelines" => "man/Transformations.md",
						 "Sampling" => "man/Sampling.md",
						 "Analysis" => "man/Analysis.md",
						 "Training" => "man/Training.md",
						 "Time Series" => "man/TimeSeries.md",
						 "Regression Models" => "man/RegressionModels.md",
						 "Regression Metrics" => "man/regressMetrics.md",
			 			 "Classification Models" => "man/ClassificationModels.md",
						 "Classification Metrics" => "man/classMetrics.md",
						 "Tree Methods" => "man/Trees.md",
						 "Ensemble Models" => "man/Ensemble.md",
						 "Model Analysis" => "man/modelanalysis.md",
						 "Plotting" => "man/Plotting.md",
						 "Clustering" => "man/Clustering.md",
						 "MultiWay" => "man/MultiWay.md",
						 "Anomaly Detection" => "man/AnomalyDetection.md",
						 "Curve Resolution" => "man/CurveResolution.md",
						 "Stats." => "man/Stats.md",
						 "Distance Measures" => "man/Dists.md",
						 "PSO" => "man/PSO.md",
						 "Genetic Algorithms" => "man/GeneticAlgorithms.md",
		    			],
			 "Full API" => "man/FullAPI.md",
		]
)

deploydocs(
    repo = "github.com/caseykneale/ChemometricsTools.jl.git",
	branch = "gh-pages",
)
