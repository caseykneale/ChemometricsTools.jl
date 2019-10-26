#push!(LOAD_PATH, "/home/caseykneale/Desktop/ChemometricsTools/ChemometricsTools.jl/");
using ChemometricsTools
#View the data in the package space
ChemometricsToolsDatasets()
#Load the data from package-space
Iris = convert(Matrix, ChemometricsToolsDataset("iris.csv") ) ;
#Seperate data from labels
X = convert( Array{Float64,2}, Iris[:, 1 : ( end - 1 ) ] );
Y = Iris[:, end ];
#One hot encode the class vector
Encoding = LabelEncoding(Y);
YHot = ColdToHot(Y, Encoding);
#Shuffle the set, because the classes are ordered
Shuffle!( X, YHot );
#Split the data into a training and hold out set
( (TrainX, TrainY), (TestX, TestY) ) = SplitByProportion( X, YHot, 0.7);

#Let the shoot out begin!
@time knn = KNN(TrainX, TrainY, "euclidean")
@time lda = LDA(TrainX, TrainY; Factors = 2)
@time ldagd = GaussianDiscriminant( lda, TrainX, TrainY; Factors = 2 )
@time mnsr = MultinomialSoftmaxRegression(TrainX, TrainY; maxiters = 2000, LearnRate = 1e-3, L2 = 1e4)
@time gnb = GaussianNaiveBayes(TrainX, TrainY)
@time ct = ClassificationTree(TrainX, TrainY;gainfn = entropy, maxdepth = 8, minbranchsize = 5)
@time rf = RandomForest(TrainX, TrainY, :classification; gainfn = entropy, trees = 100,
                        maxdepth = 10,  minbranchsize = 5,
                        samples = 0.7, maxvars = nothing)
@time simca = SIMCA(TrainX, TrainY; VarianceExplained = 0.95, Quantile = 0.999)

#Let's evaluate them on the training set then the hold out - see how they do!
models = ( knn, ldagd, mnsr, gnb, ct, rf, simca );
modelnames = ( :knn, :ldagd, :mnsr, :gnb, :ct, :rf, :SIMCA );
TrainingF1s = Dict()
for ( name, model ) in zip(modelnames, models)
    Predicted = model(TrainX)
    TrainingF1s[name] = MulticlassStats( Predicted, TrainY, Encoding )[1][:Global]["FMeasure"]
end

TrainingF1s

HoldOutF1s = Dict()
for ( name, model ) in zip(modelnames, models)
    HoldOutF1s[name] = MulticlassStats( model(TestX), TestY, Encoding )[1][:Global]["FMeasure"]
end

HoldOutF1s

#Let's check specifically the random forest classification model for the virginica class
RFStats = MulticlassStats( rf(TestX), TestY, Encoding );
RFStats[2]["virginica"]

StatsToCSVs(RFStats, "/home/caseykneale/Desktop/", "RFStats")
StatsToLaTeX(RFStats, "/home/caseykneale/Desktop/", "RFStats"; maxcolumns = 3)
