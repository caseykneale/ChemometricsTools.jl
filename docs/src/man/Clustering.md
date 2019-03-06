# Clustering API Reference

## K-means Elbow Plot Recipe
```julia
  using Plots
  ExplainedVar = []
  for K in 1:10
      km = KMeans( X, K; tolerance = 1e-14, maxiters = 1000 )
      TCSS = TotalClusterSS( km )
      WCSS = WithinClusterSS( km )
      #BCSS = BetweenClusterSS( km )
      push!(ExplainedVar, WCSS / TCSS)
  end
  scatter(ExplainedVar, title = "Elbow Plot", ylabel = "WCSS/TCSS", xlabel = "Clusters (#)", label = "K-means" )
```


## Functions

```@autodocs
Modules = [ChemometricsTools]
Pages   = ["Clustering.jl"]
```
