# ChemometricsTools.jl

*A Chemometrics Suite for Julia.*

This package offers access to essential chemometrics methods in a convenient and reliable way. It is a lightweight library written for performance and longevity. That being said, it's still a bit of a work in progress and if you find any bugs please make an issue!

## Installation:
```julia
using Pkg
Pkg.add("ChemometricsTools")
```

## Support:
This package was written in [Julia 1.0.3](https://julialang.org/) but should run fine in 1.1 or later releases. That's the beauty of from scratch code with minimal dependencies.

## Ethos
Dependencies: Only base libraries (LinearAlgebra, StatsBase, Statistics, Plots) etc will be required. This is for longevity, and to provide a fast precompilation time. As wonderful as it is that other packages exist to do some of the internal operations this one needs, we won't have to worry about a breaking change made by an external author working out the kinks in a separate package. I want this to be long-term reliable without much upkeep. I'm a busy guy working a day job; I write this to warm-up before work, and unwind afterwards.

Arrays Only: In it's current state all of the algorithms available in this package operate exclusively on 1 or 2 Arrays. To be specific, the format of input arrays should be such that the number of rows are the observations, and the number of columns are the variables. This choice was made out of convenience and my personal bias. If enough users want DataFrames, Tables, JuliaDB formats, maybe this will change.

Center-Scaling: None of the methods in this package will center and scale for you. This package won't waste your time deciding if it should auto-center/scale large chunks of data every-time you do a regression/classification.

## Why Julia?
In Julia we can do mathematics like R or Matlab (no installations/imports), but write glue code as easily as python, with the expressiveness of scala, with (often) the performance of C/C++. Multidispatch makes recycling code painless, and broadcasting allows for intuitive application of operations across collections. I'm not a soft-ware engineer, but, these things have made Julia my language of choice. Try it for a week on Julia 1.0.3, if you don't get hooked, I'd be surprised.
