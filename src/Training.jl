function Shuffle( X, Y )
    Inds = StatsBase.sample( collect(1 : size(X)[1] ), size(X)[1], replace = false )
    return( X[Inds,:], Y[Inds,:] )
end

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
forceMatrix(a) = (length(size(a)) == 1) ? reshape( a, length(a), 1 ) : a

function KFoldsValidation(K::Int, x, y)
    observations = size( x )[ 1 ]
    if floor(size( x )[ 1 ] / K) == 1
        K = observations
        println("Warning: Hold out size is 1, defaulting to Leave One Out Validation")
    end
    Remainder = observations % K
    KFoldsValidation( K, floor(size( x )[ 1 ] / K), observations, Remainder, x, forceMatrix(y))
end
LeaveOneOut(x,y) = KFoldsValidation( size( x )[ 1 ], 1, size( x )[ 1 ], 0, x, forceMatrix(y) )


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
