"""
    Unfold( Z::Array )

Unfolds a tensor into a 2-tensor.
"""
function Unfold(Z::Array)
    dimensions = size( Z )
    return reshape( Z, ( dimensions[ 1 ], prod( dimensions[ 2 : end ] ) ) )
end

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

struct MultilinearPCA
    Core::Array
    Loadings::Array
    ExplainedVariance::Float64
end

"""
    HOSVD(X; Factors = 2)

Performs multilinear PCA aka Higher Order SVD aka Tucker, etc. The number of factors decomposed
can be a scalar (repeated across all modes) or a vector/tuple for each mode.

Returns a MultilinearPCA object containing (Core Tensor, Basis Tensors, Explained Variance)

Tucker, L. R. (1964). "The extension of factor analysis to three-dimensional matrices". In N. Frederiksen and H. Gulliksen (Eds.), Contributions to Mathematical Psychology. New York: Holt, Rinehart and Winston: 109–127.
"""
function HOSVD(X; Factors = 2)
    Modes = size(X)
    Factors = (length(Factors) == 1) ? repeat([Factors], length(Modes)) : Factors
    @assert length( Factors ) == length( Modes )
    @assert all( Factors .<= Modes)
    Order = length( Modes )
    Loadings = [ zeros(Modes[N], Factors[N]) for N in 1 : length(Modes) ]
    for n in 1 : Order
        resize = (n, setdiff( 1 : Order, n) )
        unfold = reshape( Base.permutedims( X, vcat(resize[1],resize[2]) ), Modes[ resize[ 1 ] ], prod( Modes[ resize[ 2 ] ] ) )
        svdres = LinearAlgebra.svd( unfold )
        factor = Factors[n]
        Loadings[n] = svdres.U[:, 1:factor]
    end
    Core = X
    modes = 1:length(Loadings)
    #ToDo Update to use the TensorProduct method...
    for i = modes #This is a doohicky to do the tensor products...
        Core = Base.permutedims(Core, vcat(i, setdiff( modes, i ) ) )
        Un = reshape( Core , size(Core)[ 1 ], prod( size(Core)[ 2 : end ] ) )
        Core = Loadings[i]' * Un
        newshape = Tuple(vcat(collect(Factors[1:i]), collect(Modes[ ( i + 1 ) : end])))
        Core = reshape( Core,  newshape  )
    end
    return MultilinearPCA(Core, Loadings, MultiNorm( Core ) / MultiNorm( X ) )
end

"""
    HOOI( X; Factors = 1, maxiters = 100, init = :HOSVD, tolerance = 1e-9 )

Performs multiway PCA aka Higher Order SVD aka Tucker, etc via the `Higher Order Orthogonal Iteration` (HOOI) of tensors.

Returns a MultilinearPCA object containing (Core Tensor, Basis Tensors, Explained Variance)

Lieven De Lathauwer, Bart De Moor, and Joos Van-dewalle. A multilinear singular value decomposition. SIAM J. Matrix Anal. Appl., 2000.
"""
function HOOI( X; Factors = 1, maxiters = 100, init = :HOSVD, tolerance = 1e-9 )
    Size = size(X)
    Modes = length(Size)
    X_loads = []
    if init == :HOSVD
        X_loads = HOSVD( X; Factors = Factors ).Loadings
    else #Random initialization - not reccomended for reproducibility...
        X_loads = [ randn( s, Factors ) for s in Size]
    end
    LastNorm = Inf
    Norm = X_loads[1]' * Unfold(X) * foldr(kron, X_loads[reverse(2:end)])
    #Norm = MultiNorm( foldl( kron, X_loads ) )
    it = 0
    while ( it < maxiters ) && ( (MultiNorm( LastNorm .- Norm ) / MultiNorm( Norm ) ) > tolerance )
        for Mode in 1 : Modes
            NotMode = setdiff( 1:Modes, Mode )
            Core = copy(X)
            for (Collapsed, NM) in enumerate( NotMode )
                Core = TensorProduct(Core, X_loads[NM], NM - Collapsed + 1, 1;
                                        RemoveSingularModes = true )
            end
            Core = Unfold( Core )
            #Get Left Hand Singular Vectors
            X_loads[ Mode ] = LinearAlgebra.svd( Core ).U[ : , 1 : Factors ]
        end
        it += 1
        LastNorm = deepcopy( Norm )
        Norm = X_loads[1]' * Unfold(X) * foldr(kron, X_loads[reverse(2:end)])
    end
    Core = X
    for M in 1 : Modes
        Core = TensorProduct(Core, X_loads[M], 1, 1;
                                RemoveSingularModes = false )
    end
    return MultilinearPCA( Core, X_loads, MultiNorm( Core ) / MultiNorm( X ) )
end

"""
    TensorProduct(A, B, IndexA, IndexB; RemoveSingularModes = true,
                        SizeA = size(A), SizeB = size(B) )

Computes the tensor product of tensors `A` & `B` across their respective indices.

Note: This is primairily an method for internal use to the library, but feel free to
use it for your own needs.

"""
function TensorProduct(A, B, IndexA, IndexB; RemoveSingularModes = true,
                        SizeA = size(A), SizeB = size(B) )
    ModesA, ModesB   = length( SizeA ), length( SizeB )
    Ac, Bc           = copy(A), copy(B)
    UnfoldA, UnfoldB = SizeA, SizeB
    #Do we need to permute A?
    if IndexA != 1
        Swap = vcat(IndexA, setdiff( 1:ModesA, IndexA )... )
        Ac = permutedims( Ac, Swap )
        UnfoldA = Swap
        SizeA = SizeA[Swap]
    end
    #Do we need to permute B?
    if IndexB != 1
        Swap = vcat(IndexB, setdiff( 1:ModesB, IndexB )...)
        Bc = permutedims( Bc, Swap )
        UnfoldB = Swap
        SizeB = SizeB[ Swap ]
    end
    #Unfold A and B into a matrix form.
    Ac = reshape( Ac, ( SizeA[ 1 ], prod( SizeA[ 2 : end ] ) ) )
    Bc = reshape( Bc, ( SizeB[ 1 ], prod( SizeB[ 2 : end ] ) ) )
    #Multiply them
    C = Ac' * Bc
    #New shape is now prod(Ja,Ka,..Za) x prod(Jb,Kb,..Zb)
    #Need to unfold to (Ja,Ka,..Za, Jb,Kb,..Zb )
    NewShape = collect(vcat( SizeA[2:end]..., SizeB[2:end]... ))
    if RemoveSingularModes
        NewShape = NewShape[ NewShape .!= 1 ]
    end
    return reshape( C, Tuple(NewShape)... )
end

struct MultilinearPLS
    XLoadings::Array    #W
    XScores::Array      #T
    YLoadings::Array    #Q
    YScores::Array      #U
    Coefficients::Array #B
    Factors::Int        ##
end

"""
    MultilinearPLS(Y, X; Factors = minimum(size(X)) - 2,
                         tolerance = 1e-8, maxiters = 200 )

Performs a Multilinear PLS regression from `X` and `Y` tensors. The number of
`Factors`, convergence `tolerance`, and the `maxiters`(maximum iterations) may be set.

Method returns a `MultilinearPLS` object.

Notes:
    - Only Y orders < 2 are currently supported.
    - X order must be >= 2
    - X orders > 3 are currently unreviewed. Please contribute!

Bro, Rasmus. (1996), Multiway calibration. Multilinear PLS. J. Chemometrics, 10: 47-61. doi:10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C
"""
function MultilinearPLS(Y, X; Factors = minimum(size(X)) - 2,
                           tolerance = 1e-8, maxiters = 200 )
    X_size,  Y_size  = size( forceMatrix( X ) ), size( forceMatrix( Y ) )
    X_order, Y_order = length( X_size ), length( Y_size )
    @assert(Y_order < 3, "Only Y orders < 2 are currently supported.")
    @assert(X_order > 2, "Use PartialLeastSquaresRegression when X order is < 3.")
    @warn("This code algorithm has only been inspected for X orders of 3, and Y orders < 3.\n Use at your own risk, and please report any bugs!")
    X_e, Y_e        = Unfold(deepcopy(X)), Unfold(deepcopy(Y))
    T = zeros( X_size[1], Factors )                     #X Scores
    W = [ zeros( i, Factors ) for i in X_size[2:end] ]  #X Loadings
    U = zeros( Y_size[1], Factors )                     #Y Scores
    Q = [ zeros( i, Factors ) for i in Y_size[2:end] ]  #Y Loadings
    B = zeros( Factors, Factors )                       #Coefficient matrix
    iteration = 0
    for Factor in 1 : Factors
        u = [0.0]
        Weights_unfolded = [0.0]
        iteration = 0
        #Initialize Y Scores
        if (length( Q ) == 1) && (Y_size[2] == 1)
            u = Y_e
        else
            #Could extend Y decomposition to higher orders but for now consider Classic PLS2
            #mpcay = HOOI( Y_e; Factors = 1 ) #u = mpcay.Core[:] .* mpcay.Loadings[ 1 ]
            u = copy( Y_e[ :, argmax( [ (Y_e[:,col]' * Y_e[:,col])[1] for col in 1:Y_size[2]] ) ] )
        end
        change = Inf
        while (change > tolerance) && (iteration < maxiters)
            iteration += 1
            #Calculate X Contributions
            u_prime = copy( u ) #Hacky but it works
            X_project = reshape( (X_e' * u), X_size[2], prod( X_size[3:end] ) )
            mpca = HOOI( X_project; Factors = 1, tolerance = tolerance)
            for (i, w) in enumerate( mpca.Loadings )
                #99.9% sure HOOI/HOSVD scales by tensor core so no need for it here?
                W[i][:, Factor] = w
            end
            Weights_unfolded = foldr(kron, reverse(mpca.Loadings) )
            T[ :, Factor ] = X_e * Weights_unfolded
            #Calculate Y Contributions
            q = transpose( T[ :, Factor ] ) * Y_e
            q = q ./ sqrt(sum( q .^ 2))
            Q[1][:, Factor] = q[:]
            u = Y_e * q'
            U[:, Factor] = u
            #Monitor convergence by tensor norm of U
            change = sqrt( sum( (u_prime - U[:, Factor]) .^ 2 ) / sum( U[:, Factor] .^ 2 ) )
        end
        #Solve for regression coefficients and prediction intermediates
        TTT = Base.inv(T[:,1:Factor]' * T[:,1:Factor]) * T[:,1:Factor]'
        B[1:Factor,Factor] .= TTT * U[:,Factor]
        #Deflate/map residues
        Y_e = Y - (T[ :, 1:Factor] * B[1:Factor, 1:Factor] * Q[1][:, 1:Factor]')
        X_e = X_e - (T[ :, Factor ] * Weights_unfolded')
    end

    return MultilinearPLS( W, T, Q, U, B, Factors)
end

"""
    (M::MultilinearPLS)( X; Factors = M.Factors )

Applies a Multilinear PLS regression object to new `X` data with a prescribed number of
`Factors`. Method returns a matrix of the calibrated size `Y`.

Bro, Rasmus. (1996), Multiway calibration. Multilinear PLS. J. Chemometrics, 10: 47-61. doi:10.1002/(SICI)1099-128X(199601)10:1<47::AID-CEM400>3.0.CO;2-C
"""
function (M::MultilinearPLS)(X; Factors = M.Factors)
    X_size = size( forceMatrix( X ) )
    X_order = length( X_size )
    XProjections = zeros( X_size[1] , Factors )
    X_e = Unfold( copy(X) )
    for Factor in 1:Factors
        Weights_unfolded = foldr(kron, map( x -> x[:,Factor]', reverse(M.XLoadings) ) )
        XProjections[:,Factor] = X_e * Weights_unfolded'
        X_e = X_e - XProjections[:,Factor] * Weights_unfolded
    end
    F = 1:Factors
    return XProjections[ :, F ] * M.Coefficients[ F, F ] * M.YLoadings[1][ :, F ]'
end
