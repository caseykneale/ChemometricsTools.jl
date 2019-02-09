using StatsBase
using LinearAlgebra

abstract type ClassificationModel end

Threshold(yhat; level = 0.5) = map( y -> (y >= level) ? 1 : 0, yhat)
#Warning this function can allow for no class assignments...
function MulticlassThreshold(yhat; level = 0.5)
    newY = zeros(size(yhat))
    for obs in 1 : size(yhat)[1]
        (val, ind) = findmax( yhat[obs,:] )
        if val > level
            newY[ind] = val
        end
    end
    return newY
end

function HighestVote(yhat)
    return [ findmax( yhat[obs,:] )[2] for obs in 1 : size(yhat)[1]  ]
end

#Untested...
struct KNN <: ClassificationModel
    X
    Y
    DistanceType::Symbol #Can be :euclidean, :manhattan, ...
end

function ( model::KNN )( Z, K = 1 )
    DistMat = zeros( size( KNN.X )[ 1 ], size( X )[ 1 ] )
    Predictions = zeros( size( X )[ 1 ] )
    #Apply Distance Fn
    if model.DistanceType == :euclidean
        DistMat = SquareEuclideanDistance(KNN.X, Z)
    elseif model.DistanceType == :manhattan
        DistMat = ManhattanDistance(KNN.X, Z)
    end
    #Find nearest neighbors and majority vote
    for obs in 1 : size( X )[ 1 ]
        Preds = sortperm( DistMat[obs, :] )[ 1 : K ]
        Predictions[ obs ] = argmax( StatsBase.countmap( KNN.Y[ Preds ] ) )
    end

    return Predictions
end

#Untested...
struct LinearDiscriminantAnalysis <: ClassificationModel
    ClassSize
    pik
    ClassMeans
    ProjectedClassMeans
    Scores
    Loadings
    ClassCovariances
    EigenValues
end

function LinearDiscriminantAnalysis(X, Y)
    (Obs, ClassNumber) = size( X )
    Variables = size( X )[ 2 ]
    #Instantiate some variables...
    ClassMeans = zeros( ClassNumber, Variables )
    ClassSize = zeros( ClassNumber )
    WithinCovariance = zeros( Variables, Variables )
    BetweenCovariance = zeros( Variables, Variables  )
    ClassCovariance = zeros( Variables, Variables  )
    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        ClassSize[class] = sum(Members)
        ClassMeans[class,:] = StatsBase.mean(X[Members,:], dims = 1)
    end
    GlobalMean = StatsBase.mean(ClassMeans, dims = 1)

    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        #calculate the between class covariance matrix
        BetweenCovariance .+= ClassSize[class] .* ( (ClassMeans[class,:] .- GlobalMean) * (ClassMeans[class,:] .- GlobalMean)'  )
        #calculate the within class covariance matrix
        MeanCentered = X[Members,:] .- ClassMeans[class, : ]'
        WithinCovariance .+= (1.0 / (Obs - Members[class])) * ( MeanCentered' * MeanCentered  )
    end

    #Calculate the discriminant axis
    eig = LinearAlgebra.eigen(Base.inv(WithinCovariance) * BetweenCovariance)
    #Project the X data into the LDA basis
    Projected = X * eig.vectors
    #Calculate the probability density functions for each class
    pik = ClassSize ./ Obs
    YPred = zeros(Obs, ClassNumber)
    ProjClassMeans = zeros( ClassNumber, Variables )
    classcovariance = []
    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        ProjClassMeans[class, :] = StatsBase.mean(Projected[Members,:], dims = 1)
        MeanCentered = Projected[Members,:] .- ProjClassMeans[class,:]'
        push!(classcovariance, (1.0 / (Obs - ClassSize[class] )) .* ( MeanCentered' * MeanCentered  ) )
    end
    return LinearDiscriminantAnalysis(  ClassSize, pik,
                                        ClassMeans, ProjClassMeans,
                                        Projected[:,reverse(1:Variables)], eig.vectors[:,reverse(1:Variables)],
                                        classcovariance, eig.values[reverse(1:Variables)] )
end

function ( model::LinearDiscriminantAnalysis )( Z; Factors = 3 )
    ClassNumber = length(model.ClassSize)
    YHat = zeros( size(Z)[1] , ClassNumber )

    for class in 1 : ClassNumber
        Projected = Z * model.Loadings[:,1:Factors]
        MeanCentered = Projected .- model.ProjectedClassMeans[class,1:Factors]'

        scalar = (2.0 * pi)^Factors * LinearAlgebra.det(model.ClassCovariances[class][1:Factors,1:Factors])
        scalar *= 1.0 / sqrt( scalar )
        for obs in 1 : size(Z)[1]
            PDF = scalar * exp(-0.5 * MeanCentered[obs,:]' * Base.inv( model.ClassCovariances[class][1:Factors,1:Factors] ) * MeanCentered[obs,:] )
            YHat[obs, class] = model.pik[class] .* PDF
        end
    end
    return YHat
end

ExplainedVariance(LDA::LinearDiscriminantAnalysis) = LDA.Eigenvalues ./ sum(LDA.Eigenvalues)

using CSV


Raw = CSV.read(/home/caseykneale/Desktop/Spectroscopy/Data)
