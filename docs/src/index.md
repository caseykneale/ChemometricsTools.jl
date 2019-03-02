# ChemometricsTools.jl

*A Chemometrics Suite for Julia.*


This package offers access to essential chemometrics methods in a convenient and reliable way. It is a lightweight library written for performance and longevity. That being said, it's still a bit of a work in progress and if you find any bugs please make an issue!

## Support:
This package was written in [Julia 1.0](https://julialang.org/) but should run fine in 1.1 or later releases. That's the beauty of from scratch code with minimal dependencies.

## Installation:
Unfortunately this is not an official Julia package yet. Until it gets curated here's how to install it,

Git clone the repository to a directory of your choosing
```julia
using Pkg
LastDir = pwd()
cd("/your/path/here/ChemometricsTools/")
Pkg.activate(".")
Pkg.resolve()
using ChemometricsTools
cd(LastDir)
```
