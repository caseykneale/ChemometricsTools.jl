using StatsBase
using LinearAlgebra

abstract type ClassificationModel end
include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Analysis.jl")


#Untested...
struct KNN <: ClassificationModel
    X
    Y
    DistanceType::Symbol #Can be :euclidean, :manhattan, ...
end

function ( model::KNN )( Z; K = 1 )
    DistMat = zeros( size( model.X )[ 1 ], size( Z )[ 1 ] )
    Predictions = zeros( size( Z )[ 1 ] )
    #Apply Distance Fn
    if model.DistanceType == :euclidean
        DistMat = SquareEuclideanDistance(model.X, Z)
    elseif model.DistanceType == :manhattan
        DistMat = ManhattanDistance(model.X, Z)
    end
    #Find nearest neighbors and majority vote
    for obs in 1 : size( Z )[ 1 ]
        Preds = sortperm( DistMat[:, obs] )[ 1 : K ]
        Predictions[ obs ] = argmax( StatsBase.countmap( model.Y[ Preds ] ) )
    end

    return Predictions
end

function ( model::LinearDiscriminantAnalysis )( Z; Factors = 3 )
    ClassNumber = length(model.ClassSize)
    YHat = zeros( size(Z)[1] , ClassNumber )

    for class in 1 : ClassNumber
        Projected = Z * model.Loadings[:,1:Factors]
        MeanCentered = Projected .- model.ProjectedClassMeans[class,1:Factors]'

        scalar = (2.0 * pi)^Factors * LinearAlgebra.det(model.ProjectedClassCovariances[class])
        scalar *= 1.0 / sqrt( scalar )
        for obs in 1 : size(Z)[1]
            PDF = scalar * exp(-0.5 * MeanCentered[obs,:]' * Base.inv( model.ProjectedClassCovariances[class][1:Factors,1:Factors] ) * MeanCentered[obs,:] )
            YHat[obs, class] = model.pi[class] .* PDF
        end
    end
    return YHat
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Transformations.jl")
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/ClassificationMetrics.jl");
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/DistanceMeasures.jl");
#
# using CSV
# using DataFrames
# Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/MIR_Fruit_purees.csv");
# Lbls = convert.(Bool, occursin.( "NON", String.(names( Raw )) )[2:end]);
# Dump = collect(convert(Array, Raw)[:,2:end]');
# Fraud = Dump[Lbls,:];
# Legit = Dump[.!Lbls,:];
#
# Train = vcat(Fraud[1:400, :], Legit[1:200, : ] );
# snv = StandardNormalVariate(Train);
# TrainS = snv(Train);
# Train_pca = PCA(TrainS; Factors = 15);
# TrainS = Train_pca(TrainS);
#
# Test = vcat(Fraud[401:end, :], Legit[201:end, : ] );
# TestS = snv(Test);
# TestS = Train_pca(TestS);
# TrnLbl = vcat( repeat([1],400),repeat([0],200) );
# TstLbl = vcat( repeat([1],232),repeat([0],151) );
#
# Enc = LabelEncoding(TrnLbl)
# Hot = ColdToHot(TrnLbl, Enc);
# Enc.LabelCount
#
# LDA = LinearDiscriminantAnalysis(TrainS , Hot)
# LDA.EigenValues
# using Plots
# a = 0
# a = scatter(LDA.Scores[1:400,1], LDA.Scores[1:400,2]);
# scatter!(a, LDA.Scores[401:600,1], LDA.Scores[401:600,2])
#
# Voted = HighestVote(LDA(TestS ; Factors = 2)) .- 1;
# using StatsBase
# StatsBase.countmap(Voted)
#
# MulticlassStats(Voted, TstLbl, Enc)
