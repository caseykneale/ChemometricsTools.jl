"""
    BTEMobjective( a, X )
Returns the scalar BTEM objective function obtained from the linear combination vector `a` and loadings `X`.
*Note: This is not the function used in the original paper. This will be updated... it was written from memory.*
"""
function BTEMobjective( a, X )
    LinComb = a' * X
    Deriv = Scale1Norm( ScaleMinMax( FirstDerivative( LinComb ) ) )
    H = entropy( Deriv )
    #These aren't the penalties used in the original method. I don't have the paper handy...
    Negatives = LinComb .< 1e-6
    Penalty = sum( abs.( LinComb[ Negatives ] ) )
    return H + Penalty
end

"""
    BTEM(X, bands = nothing; Factors = 3, particles = 50, maxiters = 1000)
Returns a single recovered spectra from a 2-Array `X`, the selected `bands`, number of `Factors`, using a Particle Swarm Optimizer.
*Note: This is not the function used in the original paper. This will be updated... it was written from memory. Also the original method uses Simulated Annealing not PSO.*
Band-Target Entropy Minimization (BTEM):  An Advanced Method for Recovering Unknown Pure Component Spectra. Application to the FTIR Spectra of Unstable Organometallic Mixtures. Wee Chew,Effendi Widjaja, and, and Marc Garland. Organometallics 2002 21 (9), 1982-1990. DOI: 10.1021/om0108752
"""
function BTEM(X, bands = nothing; Factors = 3, particles = 50, maxiters = 1000)
    ( Obs, Vars ) = size( X )
    if isa( bands, Nothing ); bands = 1 : Vars; end
    pca = PCA( X )
    (Obs, Vars) = size(Mixture)
    if isa( bands, Nothing ); bands = 1 : Vars; end
    pca = PCA(Mixture; Factors = Factors);

    (a, score, otherparticles) = PSO(objective, Bounds(-10, 10, Factors), Bounds(-0.1, 0.1, Factors), particles;
                    tolerance = 1e-6, maxiters = maxiters,
                    InertialDecay = 0.25, PersonalWeight = 0.5, GlobalWeight = 0.5, InternalParams = pca.Loadings[:,bands]  )
    #PSO is pretty nice... It returns N other particles,
    #But whatever... Let's just return the main one... It's also a lot faster then MO-SA and easier to write...
    return a
end

"""
    NMF(X; Factors = 1, tolerance = 1e-7, maxiters = 200)
Performs a variation of non-negative matrix factorization on Array `X` and returns the a 2-Tuple of (Concentration Profile, Spectra)
*Note: This is not a coordinate descent based NMF. This is a simple fast version which works well enough for chemical signals*
Algorithms for non-negative matrix factorization. Daniel D. Lee. H. Sebastian Seung. NIPS'00 Proceedings of the 13th International Conference on Neural Information Processing Systems. 535-54
"""
function NMF(X; Factors = 1, tolerance = 1e-7, maxiters = 200)
    (Obs, Vars) = size(X)
    W = abs.( randn( Obs , Factors ) )
    H = abs.( randn( Factors, Vars ) )

    Last = zeros(Obs,Vars)
    tolerancesq = tolerance ^ 2
    iters = 0
    #Monitor change in F norm...
    while (sum( ( Last .- W * H ) .^ 2)  > tolerancesq) && (iters < maxiters)
        Last = W * H
        H .*= ( W' * X ) ./ ( W' * W * H )
        W .*= ( X * H' ) ./ ( W * H * H')
        iters += 1
    end
    return (W, H)
end


"""
    SIMPLISMA(X; Factors = 1, alpha = 0.05, includedvars = 1:size(X)[2], SecondDeriv = true)
Performs SIMPLISMA on Array `X` using either the raw spectra or the Second Derivative spectra.
alpha can be set to reduce contributions of baseline, and a list of included variables in the determination
of pure variables may also be provided.
Returns a tuple of the following form: (Concentraion Profile, Pure Spectral Estimates, Pure Variables)
W. Windig, Spectral Data Files for Self-Modeling Curve Resolution with Examples Using the SIMPLISMA Approach, Chemometrics and Intelligent Laboratory Systems, 36, 1997, 3-16.
"""
function SIMPLISMA(X; Factors = 1, alpha = 0.05, includedvars = 1:size(X)[2], SecondDeriv = true)
    Xcpy = deepcopy(X)
    X = X[:,includedvars]
    if SecondDeriv
        X = map( x -> max( x, 0.0 ), -SecondDerivative( X ) )
    end
    (obs, vars) = size(X)
    Col_Std = Statistics.std(X, dims = 1) .* sqrt( (obs - 1) / obs);
    Col_Mu = Statistics.mean(X, dims = 1);
    Robust_Col_Mu = Col_Mu .+ (alpha * reduce(max, Col_Mu) );
    Norm = sqrt.( ((Col_Std .+ Robust_Col_Mu).^ 2) .+ (Col_Mu .^ 2) )
    Normed = X ./ Norm
    normcov = (Normed' * Normed) ./ obs
    purity = Col_Std ./ Robust_Col_Mu
    purvarindex = []
    weights = zeros( vars )

    for i in 1 : (Factors+1)
       for j in 1 : vars
            if i > 1
                weights[j] = LinearAlgebra.det( normcov[ [ j; purvarindex] , [j; purvarindex ]  ] )
            else
                weights[j] = LinearAlgebra.det( normcov[ j , j ] )
            end
       end
       purity_Spec = weights .* purity'
       push!(purvarindex, argmax(purity_Spec)[1])
    end

    pureX = Xcpy[ : , includedvars[purvarindex[1:end]] ]
    purespectra = pureX \ Xcpy
    pureabundance = Xcpy / purespectra

    scale = LinearAlgebra.Diagonal(1.0 ./ sum(pureabundance, dims = 2))
    pureabundance = pureabundance * scale
    purespectra = Base.inv( scale ) * purespectra
    return (pureabundance[:,2:end], purespectra[2:end,:], includedvars[purvarindex[2:end]])
end

"""
    FNNLS( A, b; LHS = false, maxiters = 500 )
Uses an implementation of Bro et. al's Fast Non-Negative Least Squares on the matrix `A` and vector `b`.
Returns regression coefficients in the form of a vector.
Bro, R., de Jong, S. (1997) A fast non-negativity-constrained least squares algorithm. Journal of Chemometrics, 11, 393-401.
"""
function FNNLS(A, b; maxiters = 500)
    ATA = A' * A
    ATb = A' * b
    X = zeros( size( ATA )[ 2 ] )
    tolerance = eps(Float16) * reduce(max,sum(abs.(ATA), dims = 1)) * prod(size(ATb))
    P = zeros( size( ATA )[ 2 ] ) .|> Int
    R = collect( 1 : length( ATb ) ) .|> Int
    r = zeros( size( ATA )[ 2 ] )
    Inds = collect( 1 : length( ATb ) ) .|> Int
    Rinds = collect( 1 : length( ATb ) ) .|> Int

    W = ATb
    inneriters = 0
    zeroint = 0 |> Int
    while any( R .> zeroint ) && any(W[Rinds] .> tolerance)
        Rinds = Inds[ R .> zeroint ]
        j = Rinds[argmax( W[Rinds] )]
        P[j] = j ; R[j] = zeroint
        Pinds = Inds[ P .> zeroint ] ; Rinds = Inds[ R .> zeroint ]
        #Update R & P
        r[Pinds] = Base.inv( ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
        r[Rinds] .= 0.0
        while any(r[Pinds] .<= tolerance) && (inneriters < maxiters)
            Select = (r .<= tolerance) .& (P .> zeroint)
            alpha = reduce( min, X[Select] ./ ( X[Select] .- r[Select] ) )
            X .+= alpha .* ( r .- X )
            Select = Inds[(abs.(X) .< tolerance) .& ( P .!= zeroint )]
            R[Select] .= Select
            P[Select] .= zeroint
            Pinds = Inds[ P .> zeroint ] ; Rinds = Inds[ R .> zeroint ]
            r[Pinds] = Base.inv( ATA[ Pinds, Pinds ]) * ATb[ Pinds ]
            r[Rinds] .= 0.0
            inneriters += 1
        end
        X = r
        W = ATb .- (ATA * X)
    end

    return X
end

"""
    MCRALS(X, C, S = nothing; norm = (false, false), Factors = 1, maxiters = 20, nonnegative = (false, false) )
Performs Multivariate Curve Resolution using Alternating Least Squares on `X` taking initial estimates for `S` or `C`.
S or C can be constrained by their `norm`, or by nonnegativity using `nonnegative` arguments.
The number of resolved `Factors` can also be set.
Tauler, R. Izquierdo-Ridorsa, A. Casassas, E. Simultaneous analysis of several spectroscopic titrations with self-modelling curve resolution.Chemometrics and Intelligent Laboratory Systems. 18, 3, (1993), 293-300.
"""
function MCRALS(X, C, S = nothing; norm = (false, false),
                Factors = 1, maxiters = 20,
                nonnegative = (false, false) )
    @assert all( isa.( [ C , S ], Nothing ) ) == false
    lowestErr = Inf
    err = zeros(maxiters)
    D = X
    isC = isa(C, Nothing)
    isS = isa(S, Nothing)
    C = isC ? zeros(size(X)[1], Factors) : C[:, 1:Factors]
    S = isS ? zeros(Factors, size(X)[2]) : S[1:Factors, :]
    C ./= norm[1] ? sum(C, dims = 2) : 1.0
    S ./= norm[2] ? sum(S, dims = 1) : 1.0
    output = (C, S, err)
    for iter in 1 : maxiters
        if !isS
            if nonnegative[2]
                for obs in 1 : size( X )[ 1 ]
                    C[obs,:] = FNNLS(S', X[obs,:])
                end
            else
                C = X * LinearAlgebra.pinv(S)
            end
            C ./= norm[1] ? sum(C, dims = 2) : 1.0
            isC = false
            D = C * S
        end
        if !isC
            if nonnegative[1]
                for var in 1:size(X)[2]
                    S[:,var] = FNNLS(C, vec(X[:,var]))
                end
            else
                S = LinearAlgebra.pinv(C) * X
            end
            S ./= norm[2] ? sum(S, dims = 1) : 1.0
            isS = false
            D = C * S
        end
        err[iter] = sum( ( X .- D ) .^ 2 ) / prod(size(X))
        if err[iter] < lowestErr
            lowestErr = err[iter]
            output = (C, S, err)
        end
    end
    return output
end


#I believe the SIMPLISMA implementation below has errors. It's a super neat algorithm, but it does
#not display expected behaviour...

# """
#     SIMPLISMA(X; Factors = 1)
# Performs SIMPLISMA on Array `X`.
# Returns a tuple of the following form: (Concentraion Profile, Pure Spectral Estimates, Pure Variables)
# *Note: This is not the traditional SIMPLISMA algorithm presented by Willem Windig.*
# REAL-TIME WAVELET COMPRESSION AND SELF-MODELING CURVE RESOLUTION FOR ION MOBILITY SPECTROMETRY. PhD. Dissertation. 2003. Guoxiang Chen.
# """
# function SIMPLISMA(X; Factors = 1)
#     (obs, vars) = size(X)
#     PurestVar = ones(Factors) .|> Int
#     Ortho = zeros(obs, Factors)
#     SSE = StatsBase.sum(X .^ 2, dims = 1)
#     e = (SSE .- StatsBase.sum(X, dims = 1).^2) / obs #RSE/SSE
#
#     PurestVar[1] = argmax(vec(e))
#     Intensity = X[:, PurestVar[1]]
#     Ortho[:,1] = Intensity ./ sqrt( Intensity' * Intensity )#2-norm
#     for F in 2 : Factors
#         proj = sum( (Ortho[:,1:(F-1)]' * X) .^ 2, dims = 1)
#         p = vec(e .* ( 1.0 .- ( proj ./ SSE ) ) )
#         p[PurestVar[1:F]] .= -Inf
#         PurestVar[F] = argmax( p )
#         Intensity = X[ : , PurestVar[ F ] ]
#         OrthTmp = Intensity .- sum(proj * ( proj' * Intensity' ) )
#         Ortho[:,F] = OrthTmp ./ sqrt.( OrthTmp' * OrthTmp )
#     end
#
#     C = X[:, PurestVar]
#     S = LinearAlgebra.pinv(C) * X
#     magnitude = vec( sum( S .^ 2, dims = 2 ) )
#     for F in 1:Factors
#         S[F,:] = (magnitude[F] <= 1e-8) ? (S[F,:] .* 0.0) : (S[F,:] ./ sqrt(magnitude[F]))
#     end
#     S .+= StatsBase.mean(X, dims = 1)
#     return (C, S, PurestVar)
# end
