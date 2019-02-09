using StatsBase
using LinearAlgebra

abstract type ClassificationModel end


#Untested...
struct KNN <: ClassificationModel
    X
    Y
    DistanceType::Symbol #Can be :euclidean, :manhattan, ...
end

function ( model::KNN )( Z, K = 1 )
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


function LinearDiscriminantAnalysis(X, Y; Factors = 1)
    (Obs, ClassNumber) = size( Y )
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
    Projected = X * real.(eig.vectors)
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
                                        Projected[:,reverse(1:Variables)],
                                        real.(eig.vectors[:,reverse(1:Variables)]),
                                        classcovariance, real.(eig.values[reverse(1:Variables)]) )
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

#
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/ClassificationMetrics.jl")
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Transformations.jl")
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/Analysis.jl")
# include("/home/caseykneale/Desktop/Spectroscopy/chemotools/ChemometricsTools/src/DistanceMeasures.jl")
#
#
#
# using CSV
# using DataFrames
# Raw = CSV.read("/home/caseykneale/Desktop/Spectroscopy/Data/MIR_Fruit_purees.csv");
# Lbls = convert.(Bool, occursin.( "NON", String.(names( Raw )) )[2:end]);
# Dump = collect(convert(Array, Raw)[:,2:end]');
# Fraud = Dump[Lbls,:];
# Legit = Dump[.!Lbls,:];
#
# size(Fraud)
# size(Legit)
#
# Train = vcat(Fraud[1:400, :], Legit[1:200, : ] );
# snv = StandardNormalVariate(Train)
# #Train = snv(Train);
# #Train_pca = PCA(Train; Factors = 10)
# #Train = Train_pca(Train; Factors = 10);
#
#
#
# Test = vcat(Fraud[401:end, :], Legit[201:end, : ] );
# #Test = snv(Test);
# #Test = Train_pca(Test; Factors = 10);
#
# TrnLbl = vcat( repeat([1],400),repeat([0],200) );
# TstLbl = vcat( repeat([1],231),repeat([0],150) );
#
# Enc = LabelEncoding(TrnLbl)
# Hot = ColdToHot(TrnLbl, Enc);
#
#
# LDA = LinearDiscriminantAnalysis(Train , Hot)
#
# HighestVote(LDA(Test ; Factors = 4))
#
# sum(TstLbl)
#
#
#
knn=KNN(Train, HotToCold(TrnLbl, Enc), :euclidean)
#sum(knn(Test))

#knn(Test)

#MulticlassStats(knn(Test), HotToCold(TstLbl, Enc), Enc)

#
# sum(TstLbl)
