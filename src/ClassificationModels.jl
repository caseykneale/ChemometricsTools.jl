using StatsBase
using LinearAlgebra

abstract type ClassificationModel end

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

include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Analysis.jl")

#Generalized Gaussian Discriminant Analysis
struct GaussianDiscriminant
    Basis::Union{PCA, LDA}
    ClassSize
    pi
    ProjectedClassMeans
    ProjectedClassCovariances
end

function GaussianDiscriminant(M, X, Y; Factors = nothing)
    (Obs, ClassNumber) = size( Y )
    Variables = size( X )[ 2 ]
    ClassSize = zeros( ClassNumber )
    Factors = isa(Factors, Nothing) ? length(M.Values) : Factors
    Projected = M(X; Factors = Factors)
    #Calculate the probability density functions for each class
    YPred = zeros(Obs, ClassNumber)
    ProjClassMeans = zeros( ClassNumber, Factors)
    ClassCovariances = []
    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        ClassSize[class] = sum(Members)
        ProjClassMeans[class, :] = StatsBase.mean(Projected[Members,:], dims = 1)
        MeanCentered = Projected[Members,:] .- ProjClassMeans[class,:]'
        push!(ClassCovariances, (1.0 / (ClassSize[class] - 1.0 )) .* ( MeanCentered' * MeanCentered  ) )
    end
    return GaussianDiscriminant(M, ClassSize, ClassSize ./ Obs, ProjClassMeans, ClassCovariances )
end

function ( model::GaussianDiscriminant )( Z; Factors = length(model.Basis.Values) )
    ClassNumber = length(model.ClassSize)
    YHat = zeros( size(Z)[1] , ClassNumber )
    Projected = model.Basis(Z; Factors = Factors)
    for class in 1 : ClassNumber
        MeanCentered = Projected .- model.ProjectedClassMeans[class,1:Factors]'
        ProjClassCov = model.ProjectedClassCovariances[ class ][1:Factors, 1:Factors]
        scalar = 1.0 / sqrt( ( 2.0 * pi )^Factors * LinearAlgebra.det( ProjClassCov ) )
        for obs in 1 : size(Z)[1]
            PDF = scalar * exp(-0.5 * MeanCentered[obs,:]' * Base.inv( ProjClassCov ) * MeanCentered[obs,:] )
            YHat[obs, class] = model.pi[class] .* PDF
        end
    end
    return YHat
end
