"""
    MPCA(X; Factors = 2)

Performs multiway PCA aka Higher Order SVD aka Tucker, etc. The number of factors decomposed
can be a scalar(repeated across all modes) or a vector/tuple for each mode.

Returns a tuple of (Core Tensor, Basis Tensors)

ToDo: Add projection steps, maybe singular values, find multiway dataset to share...
"""
function MPCA(X; Factors = 2)
    Modes = size(X)
    Factors = (length(Factors) == 1) ? repeat([Factors], length(Modes)) : Factors
    @assert length( Factors ) == length( Modes )
    @assert all( Factors .<= reduce( min, Modes ) )
    Order = length(Modes)
    Loadings = [zeros(Modes[N], Factors[N]) for N in 1 : length(Modes) ]
    for n in 1:Order
        resize = (n, setdiff( 1 : Order, n) )
        unfold = reshape( Base.permutedims( X, vcat(resize[1],resize[2]) ), Modes[ resize[ 1 ] ], prod( Modes[ resize[ 2 ] ] ) )
        svdres = LinearAlgebra.svd(unfold)
        factor = Factors[n]
        Loadings[n] = svdres.U[:, 1:factor]
    end
    Core = X
    modes = 1:length(Loadings)
    for i = modes #This is a doohicky to do the tensor products...
        Core = Base.permutedims(Core, vcat(i, setdiff( modes, i ) ) )
        Un = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
        Core = Loadings[i]' * Un
        newshape = Tuple(vcat(collect(Factors[1:i]), collect(Modes[ ( i + 1 ) : end])))
        Core = reshape( Core,  newshape  )
    end
    Core = Base.permutedims(Core, reverse( modes ) )
    return (Core, Loadings)
end
