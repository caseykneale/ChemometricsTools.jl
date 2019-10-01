"""
    Shuffle( X, Y )

Shuffles the rows of the `X` and `Y` data without replacement. It returns a 2-Tuple of the shuffled set.
"""
function Shuffle( X, Y )
    Inds = StatsBase.sample( collect(1 : size(X)[1] ), size(X)[1], replace = false )
    return( X[Inds,:], Y[Inds,:] )
end

"""
    Shuffle!( X, Y )

Shuffles the rows of the `X` and `Y` data without replacement in place. In place, means that this function
alters the order of the data in memory and this function does not return anything.
"""
function Shuffle!( X, Y )
    Inds = StatsBase.sample( collect(1 : size(X)[1] ), size(X)[1], replace = false )
    X .= X[Inds,:]
    Y .= Y[Inds,:]
end

struct KFoldsValidation
    K::Int
    FoldSize::Int
    observations::Int
    remainder::Int
    X
    Y
end

"""
    KFoldsValidation(K::Int, x, y)

Returns a KFoldsValidation iterator with `K` folds. Because it's an iterator it can be used in for loops,
see the tutorials for pragmatic examples. The iterator returns a 2-Tuple of 2-Tuples which have the
 following form: ``` ((TrainX,TrainY),(ValidateX,ValidateY) ```.

"""
function KFoldsValidation(K::Int, x, y)
    observations = size( x )[ 1 ]
    if floor(size( x )[ 1 ] / K) == 1
        K = observations
        println("Warning: Hold out size is 1, defaulting to Leave One Out Validation")
    end
    Remainder = observations % K
    KFoldsValidation( K, floor(size( x )[ 1 ] / K), observations, Remainder, x, forceMatrix(y))
end

"""
    LeaveOneOut(x, y)

Returns a KFoldsValidation iterator with leave one out folds. Because it's an iterator it can be used in for loops,
see the tutorials for pragmatic examples. The iterator returns a 2-Tuple of 2-Tuples which have the
 following form: ``` ((TrainX,TrainY),(ValidateX,ValidateY) ```.
"""
LeaveOneOut(x, y) = KFoldsValidation( size( x )[ 1 ], 1, size( x )[ 1 ], 0, x, forceMatrix(y) )

function Base.iterate( iter::KFoldsValidation, state = (0,0) )
    ( i, o ) = state;
    predictInds = []; trainInds = []; holdoutInds = []
    r = ( ( iter.K - i) <= iter.remainder) ? 1 : 0

    if i == 0
        holdoutInds = collect( 1 : iter.FoldSize )
        trainInds = collect( ( iter.FoldSize + 1 ) : iter.observations )
    elseif i < iter.K
        HOBegin = ( i * iter.FoldSize ) + 1 + (o)
        HOEnd = ( i + 1 ) * iter.FoldSize + o + r
        TBegin = collect( 1 : (HOBegin - 1) )
        TEnd = collect( ( HOEnd + 1 ) : iter.observations )
        holdoutInds = collect( HOBegin : HOEnd )
        trainInds = [ TBegin; TEnd]
    else
        return nothing
    end
    return (    (   ( iter.X[trainInds,:]     , iter.Y[trainInds,:] ),
                    ( ( iter.X[holdoutInds,:]   , iter.Y[holdoutInds,:] ) )
                ),
                ( i + 1 , o + r) )
end
