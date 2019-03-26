#!/usr/bin/env julia

#Start Test Script
using ChemometricsTools
using Test
#Pkg.test("ChemometricsTools")
#Pkg.test()

#FNNLS tests...
# a = reshape( [73,111,52,87, 7,4, 46,72,27,80,89 , 71], 4,3)
# b = [96,7, 68,10]
# FNNLS(a, b)

# a = randn(4,4);
# b = randn(4);
# x = FNNLS( a,  b)
#
# #Torture test...
# counterrs = 0
# for i in 1:10000
#     a = randn(4,4);
#     b = randn(4);
#     x = FNNLS( a,  b)
#     if any(x .< -1e-2)
#         counterrs += 1
#     end
# end
# counterrs



@testset "Transformations" begin
    simplearray = [[1,2,3] [1,2,3]];
    Xform = Center( simplearray )
    @test all( Xform( simplearray ) .== [ [-1,0,1] [-1,0,1] ] )
end

@testset "Pipelines" begin
    #Test a longish pipeline
    FauxData1 = randn(5,10);
    PreprocessPipe = Pipeline(FauxData1, RangeNorm, Center, Scale, RangeNorm);
    Processed = PreprocessPipe(FauxData1);
    @test RMSE( FauxData1, PreprocessPipe(Processed; inverse = true) ) < 1e-14

    #Test inplace pipeline
    OriginalCopy = copy(FauxData1);
    InPlacePipe = PipelineInPlace(FauxData1, RangeNorm, Center, Scale, RangeNorm);
    @test FauxData1 != OriginalCopy
    @test Processed == FauxData1

    #Inplace transform the data back
    InPlacePipe(FauxData1; inverse = true)
    @test RMSE( OriginalCopy, FauxData1 ) < 1e-14

    #Ensure that centerscale center then scales
    Pipe1 = Pipeline(FauxData1, Center, Scale);
    Pipe2 = Pipeline(FauxData1, CenterScale);
    @test RMSE( Pipe1(FauxData1), Pipe2(FauxData1) ) < 1e-14

    #Logit can Inf out pretty easily so test with a small vector
    FauxData2 = [1,1,2,3,4,5,6,7] ./ 10.0;
    Pipe1 = Pipeline(FauxData2,  Logit);
    @test RMSE( FauxData2, Pipe1(Pipe1(FauxData2); inverse = true) ) < 1e-14

    #Test Quantile scaler
    # Pipe1 = Pipeline(FauxData1, x -> QuantileTrim(x; quantiles = (0.25,0.75)), RangeNorm);
    # Pipe1(FauxData1)
end


@testset "Find Peaks" begin
    y = sin.( collect(1:720) .* (pi/180) );
    @test findpeaks(y) == [90, 450]
    @test y[findpeaks(y)] == [1.0, 1.0]
end

@testset "Classification Metrics" begin
    CE = [ :a,:b,:c,:a,:b,:c ]
    @test IsColdEncoded( CE ) == true
    LEnc = LabelEncoding( CE )
    HOT = ColdToHot(CE, LEnc)
    @test HOT == [[1,0,0,1,0,0] [0,1,0,0,1,0] [0,0,1,0,0,1]]
    COLD = HotToCold([[1,0,0,1,0,0] [0,1,0,0,1,0] [0,0,1,0,0,1]], LEnc)
    A = MulticlassStats(COLD, CE, LEnc)
    @test A[1][:Global]["Accuracy"] == 1.0
    @test A == MulticlassStats(HOT, CE, LEnc)
    @test A == MulticlassStats(HOT, HOT, LEnc)
    @test A == MulticlassStats(CE, HOT, LEnc)
end

@testset "Distance Measures" begin
    a = [[1.0,2.0,3.0] [1.0,2.0,3.0]]
    @test all(EuclideanDistance(a) .== [[0,sqrt(2),sqrt(8)] [sqrt(2),0,sqrt(2)] [sqrt(8),sqrt(2),0]])
    @test all(ManhattanDistance(a) .== [[0.0,2.0,4.0] [2.0,0.0,2.0] [4.0,2.0,0.0]] )
end

@testset "FastNNLS Test" begin
    #Test from Paper
    a = reshape( [73,87,72,80, 71,74, 2,89,52,46,7 , 71], 4,3)
    b = [49,67, 68,20];
    c = FNNLS(a, b)
    @test all( [0.649, -1e-6, -1e-6] .< c )
    @test all( c .< [0.651,1e-6,1e-6] )
end
