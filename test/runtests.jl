#!/usr/bin/env julia

#Start Test Script
using ChemometricsTools
using Test
#Pkg.test("ChemometricsTools")
#Pkg.test()


@testset "Transformations and Pipelines" begin
    FauxData1 = randn(5,10);
    PreprocessPipe = Pipeline(FauxData1, RangeNorm, Center);
    Processed = PreprocessPipe(FauxData1);
    @test RMSE( FauxData1, PreprocessPipe(Processed; inverse = true) ) < 1e-14

    Pipe1 = Pipeline(FauxData1, Center, Scale);
    Pipe2 = Pipeline(FauxData1, CenterScale);
    @test RMSE( Pipe1(FauxData1), Pipe2(FauxData1) ) < 1e-14

end
