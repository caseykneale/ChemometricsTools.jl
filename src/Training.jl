using StatsBase

function Shuffle( X, Y )
    Inds = StatsBase.sample( collect(1 : size(X)[1] ), size(X)[1], replace = false )
    return( X[Inds,:], Y[Inds,:] )
end

function Shuffle!( X, Y )
    Inds = StatsBase.sample( collect(1 : size(X)[1] ), size(X)[1], replace = false )
    X .= X[Inds,:]
    Y .= Y[Inds,:]
end

VenetianBlinds(X) = ( X[1:2:end], X[2:2:end] )

struct KFoldsValidation
    foldsize::Int
    observations::Int
    X
    Y
end

KFoldsValidation(folds, x, y) = KFoldsValidation( Int( floor( size( x )[ 1 ] / folds) ), size( y )[ 1 ], x, y)
LeaveOneOut(x,y) = KFoldsValidation( size( x )[ 1 ], size( y )[ 1 ], x, y )

function Base.iterate( iter::KFoldsValidation, state = 0 )
    i = state;
    if (i * iter.foldsize) >= iter.observations; return nothing; end
    predictInds = []; trainInds = []

    if i == 0
        holdoutInds = collect( 1 : iter.foldsize )
        trainInds = collect( ( iter.foldsize + 1 ) : iter.observations )
    elseif (i * iter.foldsize) < iter.observations
        holdoutInds = collect( ( ( i * iter.foldsize ) + 1 ) : ( ( i + 1 ) * iter.foldsize ) )
        trainInds = [collect( i : ( i * iter.foldsize ) ); collect( ( ( ( i + 1 ) * iter.foldsize ) + 1 ) : iter.observations ) ]
    end

    return ( ((iter.X[trainInds,:], iter.Y[trainInds]), ((iter.X[holdoutInds,:], iter.Y[holdoutInds]))),  (i + 1) )
end
