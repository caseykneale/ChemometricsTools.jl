using StatsBase
using LinearAlgebra
using Statistics

abstract type ClassificationModel end

struct KNN <: ClassificationModel
    X
    Y
    DistanceType::String #Can be "euclidean", "manhattan", ...
end

function ( model::KNN )( Z; K = 1 )
    DistMat = zeros( size( model.X )[ 1 ], size( Z )[ 1 ] )
    Predictions = zeros( size( Z )[ 1 ] )
    #Apply Distance Fn
    if model.DistanceType == "euclidean"
        DistMat = SquareEuclideanDistance(model.X, Z)
    elseif model.DistanceType == "manhattan"
        DistMat = ManhattanDistance(model.X, Z)
    end
    #Find nearest neighbors and majority vote
    for obs in 1 : size( Z )[ 1 ]
        Preds = sortperm( DistMat[:, obs] )[ 1 : K ]
        Predictions[ obs ] = argmax( StatsBase.countmap( model.Y[ Preds ] ) )
    end

    return Predictions
end

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

struct LogisticRegression
    Coefficients
    Biases
    CostPerIteration
end

function softmax(x)
  exponent = exp.(x)
  return exponent ./ sum(exponent)
end

#Uses SGD to do logistic regression,,,
#BFGS might be better.. but I can add some bells and whistles here...
#IE L1 and L2 Norms. Could call Flux, but huge over-head for a 1 line inference algorithm...
#Bias term may be broken...
function MultinomialSoftmaxRegression(X, Y; LearnRate = 1e-3, maxit = 1000, L2 = 0.0)
    (Obs, ClassNumber) = size(Y)
    Vars = size(X)[2]
    Output = zeros( size( Y ) )
    #initialize weight vector(W) as zero. Most people call this theta, not W.
    W = randn(Vars, ClassNumber)
    B = zeros(1, 1)
    #For observing training error
    CostPerIt = zeros( maxit )
    for it in 1 : maxit
        for o in 1:Obs
            Output[o,:] = softmax( ( X[o,:]' * W ) .+ B )
            #Calculate the cost function - Not MSE. We are doing cross entropy loss
            CostPerIt[it] += -sum( Y[o,:] .* log.( Output[o,:] .+ 1e-6 ) ) + (L2* sum(W .^ 2))
            residualsOutput = Y[o,:] .- Output[o,:]
            #calculate changes to be applied to the weights by these gradients and update them...
            B .+= LearnRate * B * sum(residualsOutput)
            W .+= LearnRate .* (X[o,:] * residualsOutput')
            if L2 > 0.0
                W .+= LearnRate .* ( (2.0 / L2) .* W)
            end
        end
        CostPerIt[it] /= Obs
    end
    return LogisticRegression(W, B, CostPerIt)
end

function ( model::LogisticRegression )( X )
  return softmax( (X * model.Coefficients) .+ model.Biases )
end



struct GaussianNaiveBayes
    TotalSamples::Int
    classcount::Int
    Priors
    Means
    Vars
    SDs
end

function GaussianNaiveBayes(X,Y)
    (obs, vars) = size(X)
    classes = size(Y)[2]
    ClasswisePrior = sum(Y, dims = 1) ./ obs
    ClasswiseMeans = zeroes(classes, vars)
    ClasswiseVars = zeroes(classes, vars)
    #Update dictionary of classes
    for c in classes
        ClassIndices = Y[:,c] .== 1
        ClasswiseMeans[c,:] = mean(X[ClassIndices,:], dims = 1)
        ClasswiseVars[c,:] = Statistics.var(X[ClassIndices,:], dims = 1)
    end
    return GaussianNaiveBayes(TotalSamples, classes, ClasswisePrior, ClasswiseMeans, ClasswiseVars, sqrt.(ClasswiseVars))
end

Likelihood( x, mean, var, sd ) = (1.0 ./ sqrt.(2.0 * pi * sd)) .* exp.(-0.5 .* ( (x .- mean).^2.0 ./ var) )

function (gnb::GaussianNaiveBayes)(X)
    (obs, vars) = size(X)
    Predictions = zeros(obs, gnb.classcount)
    for c in 1 : gnb.classcount #Save on some log computations...
        Predictions[:,c] .= log(1.0 + gnb.Priors[c])
    end
    for o in 1 : obs, c in 1 : gnb.classcount
        Predictions[o,c] += sum( log.(1.0 .+ Likelihood( X[o,:], gnb.Means[c,:], gnb.Vars[c,:], gnb.SDs[c,:] ) ) )
    end
    return Predictions
end
