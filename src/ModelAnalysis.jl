
"""
    ExplainedVariance(PCA::PCA)

Calculates the explained variance of each singular value in a pca object.
"""
ExplainedVariance(PCA::PCA) = ( PCA.Values .^ 2 ) ./ sum( PCA.Values .^ 2 )

"""
    ExplainedVariance(lda::LDA)

Calculates the explained variance of each singular value in an LDA object.
"""
ExplainedVariance(lda::LDA) = (lda.Values .^ 2) ./ sum(lda.Values .^ 2)

"""
    ExplainedVarianceX(X,Y, pls::PartialLeastSquares)

Calculates the explained variance in `X` & `Y` of each latent variable in a PartialLeastSquares object.
"""
function ExplainedVariance( X,Y, pls::PartialLeastSquares )
    total_x_var = sum(pls.XVariance)
    total_y_var = sum(pls.YVariance)
    expvarx = zeros(Float64, pls.Factors)
    expvary = zeros(Float64, pls.Factors)
    for lv in 1:pls.Factors
        X_residuals = Statistics.var( ( pls.XScores[:,1:lv] * pls.XLoadings[:,1:lv]') .- X, dims = 1)
        expvarx[lv] = (total_x_var - sum(X_residuals)) / total_x_var
        Y_residuals = Statistics.var( ( pls.YScores[:,1:lv] * pls.YLoadings[:,1:lv]') .- Y, dims = 1)
        expvary[lv] = (total_y_var - sum(Y_residuals)) / total_y_var
    end
    return Dict( "X" => expvarx, "Y" => expvary )
end

"""
    ExplainedVarianceX(X,Y, pls::PartialLeastSquares)

Calculates the explained variance in `X` of each latent variable in a PartialLeastSquares object.
"""
function ExplainedVarianceX( X, pls::PartialLeastSquares )
    total_x_var = sum(pls.XVariance)
    expvarx = zeros(Float64, pls.Factors)
    for lv in 1:pls.Factors
        X_residuals = Statistics.var( ( pls.XScores[:,1:lv] * pls.XLoadings[:,1:lv]') .- X, dims = 1)
        expvarx[lv] = (total_x_var - sum(X_residuals)) / total_x_var
    end
    return expvarx
end

"""
    ExplainedVarianceY(Y, pls::PartialLeastSquares)

Calculates the explained variance in `Y` of each latent variable in a PartialLeastSquares object.
"""
function ExplainedVarianceY( Y, pls::PartialLeastSquares )
    total_y_var = sum(pls.YVariance)
    expvary = zeros(Float64, pls.Factors)
    for lv in 1:pls.Factors
        Y_residuals = Statistics.var( ( pls.YScores[:,1:lv] * pls.YLoadings[:,1:lv]') .- Y, dims = 1)
        expvary[lv] = (total_y_var - sum(Y_residuals)) / total_y_var
    end
    return expvary
end

"""
    Leverage(X::Array)

Calculates the leverage of samples in a `X` from the perspective of a linearly addative model.
"""
Leverage(X::Array) = LinearAlgebra.diag( X * Base.inv(transpose(X)*X)*transpose(X) )

"""
    Leverage(pca::PCA)

Calculates the leverage of samples in a `pca` object.
"""
function Leverage(pca::PCA)
    H = LinearAlgebra.diag(pca.Scores * Base.inv(pca.Scores' * pca.Scores) * pca.Scores')
    if any(H .< 1e-9)
        @warn("Negative Leverage values. Please center X matrix.")
    end
    return H
end

"""
    Leverage(pls::PartialLeastSquares)

Calculates the leverage of samples in a `pls` object.
"""
function Leverage(pls::PartialLeastSquares)
    H = diag(pls.XScores * Base.inv(pls.XScores' * pls.XScores) * pls.XScores')
    if any(H .< 1e-9)
        @warn("Negative Leverage values. Please center X & Y matrices.")
    end
    return H
end

struct Hotelling
    Lambda::Array
    Rotations::Array
    UpperLimit::Float64
end

"""
    Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)

Computes the hotelling Tsq and upper control limit cut off of a `pca` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`. Stores this to a struct which can be used for new data.

A review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere
https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
"""
function Hotelling(X, pca::PCA; Quantile = 0.05, Variance = 1.0)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum( CumVar .<= Variance )
    @assert PCs > 0
    #Truncate loadings & scores
    Scores = pca(X; Factors = PCs)
    #Hotelling Statistic: T 2 = (X^T) W Λ − 1 W ˆ T X
    #Λ ˆ = diag ( l 1 , l 2 ,.., l l )
    Lambda = sqrt.( LinearAlgebra.Diagonal( 1.0 ./ ( pca.Values[1:PCs] ) ) )
    if Obs < 100#vetted this threshold, the calculations are correct.
        Scalar = ( PCs * ((Obs^2) - 1) ) / ( Obs * ( Obs - PCs ) )
        Threshold = Scalar * quantile(Distributions.FDist( PCs, Obs - PCs ), Quantile)
    else
        Threshold =  quantile( Chisq( PCs ),  Quantile )
    end
    return Hotelling( Lambda, (Diagonal( 1.0 ./ pca.Values[1:PCs]) * pca.Loadings[1:PCs,:])', Threshold)
end

"""
    Hotelling(X, PLS::PartialLeastSquares; Quantile = 0.05, LVs = 1)

Computes the hotelling Tsq and upper control limit cut off of a `PartialLeastSquares` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`. Stores this to a struct which can be used for new data.

Note: The number of latent variables cannot be automatically set by the explained variance in X. It is not computed cumulatively.

Informative PLS score-loading plots for processunderstanding and monitoring. Rolf Ergon.
Journal of Process Control 14 (2004) 889-897
https://pdfs.semanticscholar.org/89b6/677a592dbe05a9b754d377ade416d7a17393.pdf
"""
function Hotelling(X, pls::PartialLeastSquares; Quantile = 0.05, LVs = 1)
    (Obs,Vars) = size(X)
    @assert LVs > 0
    #Truncate loadings & scores
    normloads = pls.XLoadings[:,1:LVs] ./ sum( pls.XLoadings[:,1:LVs] .^ 2, dims = 1) ;
    Scores = X * normloads #* pls.RWeights[ : , 1:LVs ]
    #Hotelling Statistic: T 2 = (X^T) W Λ − 1 W ˆ T X
    #Λ ˆ = diag ( l 1 , l 2 ,.., l l )
    Lambda = LinearAlgebra.Diagonal( 1. ./ LinearAlgebra.diag( Scores' * Scores )  )
    if Obs < 100
        Scalar = ( LVs * ((Obs^2) - 1) ) / ( Obs * ( Obs - LVs ) )
        Threshold = Scalar * quantile(Distributions.FDist( LVs, Obs - LVs ), Quantile)
    else
        Threshold = quantile( Chisq( LVs ),  Quantile )
    end
    return Hotelling( Lambda, normloads, Threshold )
end

"""
    (H::Hotelling)(X)

Retrieves the `T^2` statistic from a saved Hotelling model.
Note 1: This does not automatically center or scale `X`.
Note 2: if the model used to generate the Hotelling struct changes, so will the Hotelling struct(pass by reference).

"""
function (H::Hotelling)(X)
    Scores = X * H.Rotations
    return diag( Scores * H.Lambda * Scores' )
end

struct Q
    Rotations::Array
    Projection::Array
    UpperLimit::Float64
end

"""
    Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)

Computes the Q-statistic and upper control limit cut off of a `pca` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`.

A review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere
https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
"""
function Q(X, pca::PCA; Quantile = 0.95, Variance = 1.0)
    (Obs,Vars) = size(X)
    CumVar = cumsum( ExplainedVariance( pca ) )
    PCs = sum(CumVar .<= Variance)
    #Q = diag(X * (LinearAlgebra.Diagonal(ones(Vars)) - pca.Loadings[1:PCs,:]' * pca.Loadings[1:PCs,:]) * X')
    L = [ sum(pca.Values[(PCs + 1) : end] .^ order) for order in [ 1, 2, 3 ] ]
    H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
    if H0 < 1e-5
        H0 = 1e-5
    end
    Gauss = quantile( Normal(),  Quantile )
    FirstTerm = ( Gauss * sqrt( 2.0 * L[2] * (H0^2.0) ) ) / L[1]
    SecondTerm =  (L[2] * H0 * ( 1.0 - H0 ) ) / (L[1] ^ 2.0)
    Upper = L[1] * ( FirstTerm + 1.0 + SecondTerm ) ^ 2.0
    return Q( pca.Loadings[1:PCs,:]',
              LinearAlgebra.I - pca.Loadings[1:PCs,:]' * pca.Loadings[1:PCs,:],
              Upper )
end

"""
    Q(X, pls::PartialLeastSquares; Quantile = 0.95, Variance = 1.0)

Computes the Q-statistic and upper control limit cut off of a `pca` object using a specified `Quantile` and
cumulative variance explained `Variance` for new or old data `X`.

Note: The number of latent variables cannot be automatically set by the explained variance in X. It is not computed cumulatively.

A review of PCA-based statistical process monitoring methodsfor time-dependent, high-dimensional data. Bart De Ketelaere
https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf
"""
function Q(X, pls::PartialLeastSquares; Quantile = 0.95,  LVs = 1)
    (Obs,Vars) = size(X)
    normloads = pls.XLoadings[:,1:LVs] ./ sum( pls.XLoadings[:,1:LVs] .^ 2, dims = 1);
    proj = X * normloads * normloads'
    resids = proj .- X
    #Q = LinearAlgebra.diag(resids * (LinearAlgebra.I - normloads * normloads') * resids')
    eigvals = LinearAlgebra.svd( resids; full = false ).S #.^ 2
    L = [ sum(eigvals[(LVs + 1) : end] .^ order) for order in [ 1, 2, 3 ] ]
    H0 = 1.0 - ( ( 2.0 * L[ 1 ] * L[ 3 ] ) / ( 3.0 * L[ 2 ]^2 ) )
    if H0 < 1e-5
        H0 = 1e-5
    end
    Gauss = quantile( Normal(),  Quantile )
    FirstTerm = ( Gauss * sqrt( 2.0 * L[2] * (H0^2.0) ) ) / L[1]
    SecondTerm =  (L[2] * H0 * ( 1.0 - H0 ) ) / (L[1] ^ 2.0)
    Upper = L[1] * ( FirstTerm + 1.0 + SecondTerm ) ^ 2.0
    return Q(normloads, LinearAlgebra.I - normloads * normloads', Upper)
end

function (qr::Q)(X)
    proj = X * qr.Rotations * qr.Rotations'
    resids = proj .- X
    return LinearAlgebra.diag(resids * qr.Projection * resids')
end
