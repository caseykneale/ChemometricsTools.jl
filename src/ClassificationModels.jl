abstract type ClassificationModel end

"""
    KNN( X, Y; DistanceType::String )

DistanceType can be "euclidean", "manhattan". `Y` Must be one hot encoded.

Returns a KNN classification model.
"""
struct KNN <: ClassificationModel
    X::Array{Float64,2}
    Y::Array{Float64,2}
    DistanceType::String
end

"""
    ( model::KNN )( Z; K = 1 )

Returns a 1 hot encoded inference from `X` with `K` Nearest Neighbors, using a KNN object.
"""
function ( model::KNN )( Z; K = 1 )
    MostCommon = 0
    Obs = size( Z )[ 1 ]
    Classes = size(model.Y)[2]
    DistMat = zeros( size( model.X )[ 1 ], Obs )
    #Predictions = zeros( size( Z )[ 1 ] )
    Predictions = zeros( Obs, Classes )
    #Apply Distance Fn
    if model.DistanceType == "euclidean"
        DistMat = SquareEuclideanDistance(model.X, Z)
    elseif model.DistanceType == "manhattan"
        DistMat = ManhattanDistance(model.X, Z)
    end
    #Find nearest neighbors and majority vote
    for obs in 1 : Obs
        Preds = sortperm( DistMat[:, obs] )[ 1 : K ]
        if K == 1
            lbls = argmax(model.Y[ Preds,: ], dims = 2 )
            MostCommon = argmax( StatsBase.countmap( lbls ) )[2]
        else
            lbls = argmax(model.Y[ Preds,: ], dims = 1 )
            MostCommon = argmax( StatsBase.countmap( lbls ) )[2]
        end
        Predictions[ obs, MostCommon ] =  1
    end
    return Predictions
end
#Generalized Gaussian Discriminant Analysis
struct GaussianDiscriminant
    Basis::Union{PCA, LDA}
    ClassSize::Array
    pi::Array
    ProjectedClassMeans::Array{Float64,2}
    ProjectedClassCovariances::Array
end

"""
    GaussianDiscriminant(M, X, Y; Factors = nothing)

Returns a GaussianDiscriminant classification model on basis object `M` (PCA, LDA) and one hot encoded `Y`.
"""
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

"""
    ( model::GaussianDiscriminant )( Z; Factors = size(model.ProjectedClassMeans)[2] )

Returns a 1 hot encoded inference from `Z` using a GaussianDiscriminant object.
"""
function ( model::GaussianDiscriminant )( Z; Factors = size(model.ProjectedClassMeans)[2] )
    MaximumLatentFactors = size(model.ProjectedClassMeans)[2]
    @assert Factors <= MaximumLatentFactors
    ClassNumber = length(model.ClassSize)
    YHat = zeros( size(Z)[1] , ClassNumber )
    Projected = model.Basis(Z; Factors = Factors)

    for class in 1 : ClassNumber
        MeanCentered = Projected .- model.ProjectedClassMeans[class, 1:Factors]'
        ProjClassCov = model.ProjectedClassCovariances[ class ][1:Factors, 1:Factors]
        scalar = 1.0 / sqrt( ( 2.0 * pi )^Factors * LinearAlgebra.det( ProjClassCov ) + 1e-9)
        for obs in 1 : size(Z)[1]
            PDF = scalar * exp(-0.5 * MeanCentered[obs,:]' * Base.inv( ProjClassCov ) * MeanCentered[obs,:] )
            YHat[obs, class] = model.pi[class] .* PDF
        end
    end
    return YHat
end

struct LogisticRegression
    Coefficients
    Biases
    CostPerIteration
end

function softmax(x)
  exponent = exp.(x)
  return exponent ./ sum(exponent)
end

"""
    MultinomialSoftmaxRegression(X, Y; LearnRate = 1e-3, maxiters = 1000, L2 = 0.0)

Returns a LogisticRegression classification model made by Stochastic Gradient Descent.
"""
function MultinomialSoftmaxRegression(X, Y; LearnRate = 1e-3, maxiters = 1000, L2 = 0.0)
    (Obs, ClassNumber) = size(Y)
    Vars = size(X)[2]
    Output = zeros( size( Y ) )
    #initialize weight vector(W) as zero. Most people call this theta, not W.
    W = randn(Vars, ClassNumber)
    B = zeros(1, 1)
    #For observing training error
    CostPerIt = zeros( maxiters )
    for it in 1 : maxiters
        for o in 1:Obs
            Output[o,:] = softmax( ( X[o,:]' * W ) .+ B )
            #Calculate the cost function - Not MSE. We are doing cross entropy loss
            CostPerIt[it] += -sum( Y[o,:] .* log.( Output[o,:] .+ 1e-6 ) ) + (L2 * sum(W .^ 2))
            residualsOutput = Y[o,:] .- Output[o,:]
            #calculate changes to be applied to the weights by these gradients and update them...
            B .+= LearnRate * [1.0] * sum(residualsOutput)
            W .+= LearnRate .* (X[o,:] * residualsOutput')
            if L2 > 0.0
                W .+= LearnRate .* ( (2.0 / L2) .* W)
            end
        end
        CostPerIt[it] /= Obs
    end
    return LogisticRegression(W, B, CostPerIt)
end

"""
    ( model::LogisticRegression )( X )

Returns a 1 hot encoded inference from `X` using a LogisticRegression object.
"""
( model::LogisticRegression )( X ) = softmax( (X * model.Coefficients) .+ model.Biases )


struct GaussianNaiveBayes
    TotalSamples::Int
    classcount::Int
    Priors
    Means::Array{Float64,2}
    Vars::Array{Float64,2}
end

"""
    GaussianNaiveBayes(X,Y)

Returns a GaussianNaiveBayes classification model object from `X` and one hot encoded `Y`.
"""
function GaussianNaiveBayes(X,Y)
    (obs, vars) = size(X)
    classes = size(Y)[2]
    ClasswisePrior = sum(Y, dims = 1) ./ obs
    ClasswiseMeans = zeros(classes, vars)
    ClasswiseVars = zeros(classes, vars)
    for c in 1 : classes
        ClassIndices = Y[:,c] .== 1
        ClasswiseMeans[c,:] = mean(X[ClassIndices,:], dims = 1)
        ClasswiseVars[c,:] = Statistics.var(X[ClassIndices,:], dims = 1)
    end
    return GaussianNaiveBayes(obs, classes, ClasswisePrior, ClasswiseMeans, ClasswiseVars)
end

Likelihood( x, mean, var ) = ( 1.0 ./ sqrt.( 2.0 * pi * var ) ) .* exp.( -0.5 .* ( ( (x .- mean) .^ 2.0 ) ./ var ) )

"""
    (gnb::GaussianNaiveBayes)(X)

Returns a 1 hot encoded inference from `X` using a GaussianNaiveBayes object.
"""
function (gnb::GaussianNaiveBayes)(X)
    (obs, vars) = size(X)
    Predictions = zeros(obs, gnb.classcount)
    for o in 1 : obs, c in 1 : gnb.classcount
        Predictions[o,c] = gnb.Priors[c] * prod( Likelihood( X[o,:], gnb.Means[c,:], gnb.Vars[c,:] ) )
    end
    return Predictions
end
