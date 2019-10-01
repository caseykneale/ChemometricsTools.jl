struct MultiCenter{B} <: Transform
    Mean::B
    Mode::Int
    invertible::Bool
end

"""
    MultiCenter(Z, mode = 1)

Acquires the mean of the specified mode in `Z` and returns a transform that will remove those means from any future data.
"""
function MultiCenter(Z, mode = 1)
    Modes = size(Z)
    Core = Base.permutedims(Z, vcat(mode, setdiff( 1:length(Modes), mode ) ) )
    Core = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
    return MultiCenter( StatsBase.mean(Core, dims = 2), mode, true )
end

"""
    (T::MultiCenter)(Z; inverse = false)

Centers data in Tensor `Z` mode-wise according to learned centers in MultiCenter object `T`.
"""
function (T::MultiCenter)(Z; inverse = false)
    Modes = size(Z)
    Core = Base.permutedims(Z, vcat(T.Mode, setdiff( 1:length(Modes), T.Mode ) ) )
    ModesPerm = size(Core)
    Core = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
    if inverse
        Core = Core .+ T.Mean
    else
        Core = Core .- T.Mean
    end
    Core = reshape( Core,  ModesPerm  )
    Revert = collect(2:length(Modes))
    splice!(Revert, T.Mode:(T.Mode - 1), [1])
    return Base.permutedims(Core, Revert )
end

struct MultiScale{B} <: Transform
    StdDev::B
    Mode::Int
    invertible::Bool
end

"""
    MultiScale(Z, mode = 1)

Acquires the standard deviations of the specified mode in `Z` and returns a transform that will scale by those standard deviations from any future data.
"""
function MultiScale(Z, mode = 1)
    Modes = size(Z)
    Core = Base.permutedims(Z, vcat(mode, setdiff( 1:length(Modes), mode ) ) )
    Core = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
    return MultiCenter( StatsBase.std(Core, dims = 2), mode, true )
end

"""
    (T::MultiScale)(Z; inverse = false)

Scales data in Tensor `Z` mode-wise according to learned standard deviations in MultiScale object `T`.
"""
function (T::MultiScale)(Z; inverse = false)
    Modes = size(Z)
    Core = Base.permutedims(Z, vcat(T.Mode, setdiff( 1:length(Modes), T.Mode ) ) )
    ModesPerm = size(Core)
    Core = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
    if inverse
        Core = Core .* T.StdDev
    else
        Core = Core ./ T.StdDev
    end
    Core = reshape( Core,  ModesPerm  )
    Revert = collect(2:length(Modes))
    splice!(Revert, T.Mode:(T.Mode - 1), [1])
    return Base.permutedims(Core, Revert )
end

"""
    MultiNorm(T)

Computes the equivalent of the Froebinius norm on a tensor `T`. Returns a scalar.
"""
MultiNorm(T) = sqrt(sum(T .^ 2))

"""
    MultiPCA(X; Factors = 2)

Performs multiway PCA aka Higher Order SVD aka Tucker, etc. The number of factors decomposed
can be a scalar(repeated across all modes) or a vector/tuple for each mode.

Returns a tuple of (Core Tensor, Basis Tensors)
"""
function MultiPCA(X; Factors = 2)
    Modes = size(X)
    Factors = (length(Factors) == 1) ? repeat([Factors], length(Modes)) : Factors
    @assert length( Factors ) == length( Modes )
    @assert all( Factors .<= Modes)
    Order = length(Modes)
    Loadings = [zeros(Modes[N], Factors[N]) for N in 1 : length(Modes) ]
    for n in 1 : Order
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
    return (Core, Loadings, MultiNorm(Core) / MultiNorm(X))
end
