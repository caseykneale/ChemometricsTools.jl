"""
    SplitByProportion(X::Array, Proportion::Float64 = 0.5)

Splits `X` Array along the observations dimension into a 2-Tuple based on the `Proportion`.
The form of the output is the following: ( X1, X2 )
"""
function SplitByProportion(X::Array, Proportion::Float64 = 0.5)
    FirstChunk = Int(floor(size(X)[1] * Proportion))
    return ( (X[ 1:FirstChunk, : ] ), (X[ ( FirstChunk + 1 ):end, : ] ) )
end

"""
    SplitByProportion(X::Array, Y::Array,Proportion::Float64 = 0.5)

Splits an `X` and Associated `Y` Array along the observations dimension into a 2-Tuple of 2-Tuples based on the `Proportion`.
The form of the output is the following: ( (X1, Y1), (X2, Y2) )
"""
function SplitByProportion(X::Array, Y::Array, Proportion::Float64 = 0.5)
    Y = forceMatrix(Y)
    FirstChunk = Int(floor(size(X)[1] * Proportion))
    return ((X[1:FirstChunk,:], Y[1:FirstChunk,:]), (X[(FirstChunk+1):end,:], Y[(FirstChunk+1):end,:])   )
end

"""
    VenetianBlinds(X)

Splits an `X` Array along the observations dimension into a 2-Tuple of 2-Tuples based on the whether it is even or odd.
The form of the output is the following: ( X1, X2 )
"""
VenetianBlinds(X) = ( X[1:2:end,:], X[2:2:end,:] )
"""
    VenetianBlinds(X,Y)

Splits an `X` and associated `Y` Array along the observation dimension into a 2-Tuple of 2-Tuples based on the whether it is even or odd.
The form of the output is the following: ( (X1,Y1), (X2, Y2) )
"""
VenetianBlinds(X,Y) = (( X[1:2:end,:], Y[1:2:end,:] ), ( X[2:2:end,:], Y[2:2:end,:] ) )

CICol(a::CartesianIndex) = collect(Tuple(a))[2]

CIRow(a::CartesianIndex) = collect(Tuple(a))[1]

"""
    KennardStone(X, TrainSamples; distance = "euclidean")

Returns the indices of the Kennard-Stone sampled exemplars (E), and those not sampled (O) as a 2-Tuple (E, O).

R. W. Kennard & L. A. Stone (1969) Computer Aided Design of Experiments, Technometrics, 111, 137-148, DOI: 10.1080/00401706.1969.10490666
"""
function KennardStone(X, TrainSamples; distance = "euclidean")
    Obs = 1:size(X)[1]
    FullSet = collect( Obs )
    DistMat = []
    if distance == "euclidean"
        DistMat = SquareEuclideanDistance( X )
    elseif distance == "manhattan"
        DistMat = ManhattanDistance( X )
    end
    @assert DistMat != []
    DistMat = hcat([[0] ; Obs], vcat( Obs', DistMat ))
    Candidates = Array{Int64, 1}(zeros( TrainSamples ))
    #Grab the 2 most seperated points
    Candidates[ 1 : 2 ] =  collect(Tuple(argmax( DistMat[ 2 : end, 2 : end ] ))) .|> Int64
    Candidates[ 1 : 2 ] .+= 1
    Available = setdiff( FullSet, Candidates[1:2])
    for sample in 3 : TrainSamples
        CurrentView = DistMat[ [ [ 1 ];   Available ],  [ [ 1 ];  Candidates[ 1:(sample - 1)  ]  ] ]
        #Find nearest observation(column) to each of the points that were selected(row)
        NearestNeighbors = argmin( CurrentView[2:end, 2:end], dims = 2)
        Values = CurrentView[2:end, 2:end][NearestNeighbors]#Get those values...
        FarNeighbor = argmax( Values )#Find the maximum
        #Get the column these values were associated with
        PointRelativeIndex = CurrentView[ CIRow(NearestNeighbors[ FarNeighbor ]) + 1, 1 ] |> Int64
        Candidates[ sample ] = PointRelativeIndex + 1
        Available = setdiff( FullSet, Candidates[ 1 : sample ]  ) #.- 1 )
    end
    return ( Candidates, Available )
end
