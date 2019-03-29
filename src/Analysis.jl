struct PCA
    Scores::Array{Float64,2}
    Loadings::Array{Float64,2}
    Values::Array
    algorithm::String
end

"""
    PCA_NIPALS(X; Factors = minimum(size(X)) - 1, tolerance = 1e-7, maxiters = 200)

Compute's a PCA from `x` using the NIPALS algorithm with a user specified number of latent variables(`Factors`).
The tolerance is the minimum change in the F norm before ceasing execution. Returns a PCA object.

"""
function PCA_NIPALS(X; Factors = minimum(size(X)) - 1, tolerance = 1e-7, maxiters = 200)
    tolsq = tolerance * tolerance
    #Instantiate some variables up front for performance...
    Xsize = size(X)
    Tm = zeros( ( Xsize[1], Factors ) )
    Pm = zeros( ( Factors, Xsize[2] ) )
    t = zeros( ( 1, Xsize[1] ) )
    p = zeros( ( 1, Xsize[2] ) )
    #Set tolerance to floating point precision
    Residuals = copy(X)
    for factor in 1:Factors
        lastErr = sum(abs.(Residuals)); curErr = tolerance + 1;
        t = Residuals[:, 1]
        iterations = 0
        while (abs(curErr - lastErr) > tolsq) && (iterations < maxiters)
            p = Residuals' * t
            p = p ./ sqrt.( p' * p )
            t = Residuals * p
            #Track change in Frobenius norm
            lastErr = curErr
            curErr = sqrt(sum( ( Residuals - ( t * p' ) ) .^ 2))
            iterations += 1
        end
        Residuals -= t * p'
        Tm[:,factor] = t
        Pm[factor,:] = p
    end
    #Find singular values/eigenvalues
    EigVal = sqrt.( LinearAlgebra.diag( Tm' * Tm ) )
    #Scale loadings by singular values
    Tm = Tm * LinearAlgebra.Diagonal( 1.0 / EigVal )
    return PCA(Tm, Pm, EigVal, "NIPALS")
end

#SVD based PCA

"""
    PCA(X; Factors = minimum(size(X)) - 1)

Compute's a PCA from `x` using LinearAlgebra's SVD algorithm with a user specified number of latent variables(`Factors`).
Returns a PCA object.

"""
function PCA(Z; Factors = minimum(size(Z)) - 1)
    svdres = LinearAlgebra.svd(Z)
    return PCA(svdres.U[:, 1:Factors], svdres.Vt[1:Factors, :], svdres.S[1:Factors], "SVD")
end

"""
    (T::PCA)(Z::Array; Factors = length(T.Values), inverse = false)

Calling a PCA object on new data brings the new data `Z` into or out of (`inverse` = true) the PCA basis.

"""
(T::PCA)(Z::Array; Factors = length(T.Values), inverse = false) = (inverse) ? Z * (Diagonal(T.Values[1:Factors]) * T.Loadings[1:Factors,:]) : Z * (Diagonal( 1.0 ./ T.Values[1:Factors]) * T.Loadings[1:Factors,:])'

"""
    ExplainedVariance(PCA::PCA)

Calculates the explained variance of each singular value in a pca object.

"""
ExplainedVariance(PCA::PCA) = ( PCA.Values .^ 2 ) ./ sum( PCA.Values .^ 2 )

struct LDA
    Scores::Array{Float64,2}
    Loadings::Array{Float64,2}
    Values::Array
end


"""
    LDA(X, Y; Factors = 1)

Compute's a LinearDiscriminantAnalysis transform from `x` with a user specified number of latent variables(`Factors`).
Returns an LDA object.

"""
function LDA(X, Y; Factors = 1)
    (Obs, ClassNumber) = size( Y )
    @assert Factors < ClassNumber
    Variables = size( X )[ 2 ]
    #Instantiate some variables...
    ClassMeans = zeros( ClassNumber, Variables )
    ClassSize = zeros( ClassNumber )
    WithinCovariance = zeros( Variables, Variables )
    BetweenCovariance = zeros( Variables, Variables  )
    ClassCovariance = zeros( Variables, Variables  )
    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        ClassSize[class] = sum(Members)
        ClassMeans[class,:] = StatsBase.mean(X[Members,:], dims = 1)
    end
    GlobalMean = StatsBase.mean(X, dims = 1)

    for class in 1 : ClassNumber
        Members = Y[ :, class ] .== 1
        #calculate the between class covariance matrix
        Diff = ClassMeans[class,:] .- GlobalMean'
        BetweenCovariance .+= ClassSize[class] .* ( Diff * Diff' )
        #calculate the within class covariance matrix
        for member in findall(Members .== true)
            MeanCentered = X[member,:] .- ClassMeans[class, : ]
            WithinCovariance .+= ( MeanCentered * MeanCentered' )
        end
    end
    #Calculate the discriminant axis'
    #eig = LinearAlgebra.eigen(Base.inv(WithinCovariance) * BetweenCovariance)
    eig = LinearAlgebra.eigen(WithinCovariance \ BetweenCovariance)
    if any( imag.( eig.values ) .> 1e-1)
        println("Warning: Some eigenvalues found to have complex contributions > 0.1")
    end
    #Maybe reccomend to the user to do pca first or centerscale or both?
    ReVals = real.(eig.values)
    Sorted = sortperm( ReVals, rev = true)
    Contributions = ReVals[Sorted] .>= 1e-9
    Loadings = real.(eig.vectors[:,Sorted[ Contributions ][1:Factors] ] )
    #Project the X data into the LDA basis
    Scores = X * Loadings
    return LDA( Scores, Loadings, ReVals[ Sorted[ Contributions][1:Factors] ])
end

"""
    ( model::LDA )( Z; Factors = length(model.Values) )

Calling a LDA object on new data brings the new data `Z` into the LDA basis.

"""
function ( model::LDA )( Z; Factors = length(model.Values) )
     Projected = Z * model.Loadings[:,1:Factors]
end

"""
    ExplainedVariance(lda::LDA)

Calculates the explained variance of each singular value in an LDA object.

"""
ExplainedVariance(lda::LDA) = (lda.Values .^ 2) ./ sum(lda.Values .^ 2)

function MatrixInverseSqrt(X, threshold = 1e-6)
    eig = eigen(X)
    diagelems = 1.0 ./ sqrt.( max.( eig.values , 0.0 ) )
    diagelems[ diagelems .== Inf ] .= 0.0
    return eig.vectors * LinearAlgebra.Diagonal( diagelems ) * Base.inv( eig.vectors )
end

#Untested...
struct CanonicalCorrelationAnalysis
    U
    V
    r
end

"""
    CanonicalCorrelationAnalysis(A, B)

Returns a CanonicalCorrelationAnalysis object which contains (U, V, r) from Arrays A and B.
"""
function CanonicalCorrelationAnalysis(A, B)
    (Obs,Vars) = size(A);;
    CAA = (1/Obs) .* A * A'
    CBB = (1/Obs) .* B * B'
    CAB = (1/Obs) .* A * B'
    maxrank = min( LinearAlgebra.rank( A ), LinearAlgebra.rank( B ) )
    CAAInvSqrt = MatrixInverseSqrt(CAA)
    CBBInvSqrt = MatrixInverseSqrt(CBB)
    singvaldecomp = LinearAlgebra.svd( CAAInvSqrt * CAB * CBBInvSqrt )
    Aprime = CAAInvSqrt * singvaldecomp.U[ :,1 : maxrank ]
    Bprime = CAAInvSqrt * singvaldecomp.Vt[ :,1 : maxrank ]
    return CanonicalCorrelationAnalysis(Aprime' * A, Bprime' * B, singvaldecomp.S[1 : maxrank] )
end

"""
    findpeaks( vY; m = 3)

Finds the indices of peaks in a vector vY with a window span of `2m`.
Original R function by Stas_G:(https://stats.stackexchange.com/questions/22974/how-to-find-local-peaks-valleys-in-a-series-of-data)
This version is based on a C++ variant by me.
"""
function findpeaks( vY; m = 3)
    @assert length(size(vY)) == 1
    sze = size(vY)[1];
    (i,q) = (0,0);#generic iterator, second generic iterator
    (lb,rb) = (0,0);#left bound, right bound
    ret = [];
    for i in 1 : ( sze - 2 )
        #Find all regions with negative laplacian between neighbors
        if (sign( vY[ i + 2 ]  - vY[ i + 1 ] ) - sign( vY[ i + 1 ]  - vY[ i ] ) ) < 0
            #Now assess all regions with negative laplacian between neighbors...
            lb = i - m - 1;# define left bound of vector
            lb = (lb < 1) ? 1 : lb
            rb = i + m + 1# define right bound of vector
            rb = (rb >= (sze-2)) ? rb = (sze-3) : rb
            #We have found a peak by our criterion
            if !any(vY[lb:rb] .> vY[i+1])
                push!( ret, i + 1 );
            end #End if found peak
        end #End if laplace condition
    end #End loop
    return ret
end



function RecursiveAlignment(aligned, reference; maxlags::Int = 800, lookahead::Int = 0, minlength::Int = 20, mincorr::Float64 = 0.05)
    if length(aligned) < minlength
        return aligned;
    end
    #Calculate cross correlation between the spectrum and the reference
    lagspan = min( length( aligned ) - 1, maxlags )
    r = StatsBase.crosscor( aligned, reference, -lagspan : lagspan  )
    maxi = argmax( r ) #Find largest value
    if r[ maxi ] <= mincorr # is the largest correlation junk?
        return aligned
    end
    lag = (maxi > lagspan/2) ? maxi - lagspan - 1 : maxi - 1;

    if lag == 0
        if lookahead <= 0
            return aligned
        end
        lookahead -= 1
    end

    #Pad spectra based on lags
    if !( ( lag == 0 ) || ( lag >= length( aligned ) ) )
        if lag > 0
    	   aligned = vcat( ones( lag ) * aligned[ 1 ], aligned[ 1 : (length(aligned) - lag) ] );
        elseif lag < 0
    	   lag = abs( lag );
    	   aligned = vcat( aligned[ ( lag + 1 ) : length( aligned ) ], ones( lag ) * aligned[ end ] );
        end
    end
    #Find new branch point
    middle = round( length( aligned ) / 2) |> Int
    quarter = floor( middle / 4 ) |> Int
    mini = argmin( aligned[ (middle - quarter ) : ( middle + quarter ) ] ) |> Int
    midpnt = mini + middle - quarter
    #Calculate branched components
    LHS = RecursiveAlignment(aligned[ 1 : midpnt ], reference[ 1 : midpnt ];
                        maxlags = maxlags, lookahead = lookahead, minlength = minlength, mincorr = mincorr);
    RHS = RecursiveAlignment(aligned[ (midpnt + 1) : end ], reference[ (midpnt + 1) : end ];
                        maxlags = maxlags, lookahead = lookahead, minlength = minlength, mincorr = mincorr);
    return vcat( LHS, RHS )
end

"""
    RAFFT(raw, reference; maxlags::Int = 500, lookahead::Int = 1, minlength::Int = 20, mincorr::Float64 = 0.05)

RAFFT corrects shifts in the `raw` spectral bands to be similar to those in a given `reference` spectra through
the use of "recursive alignment by FFT". It returns an array of corrected spectra/chromatograms. The number of maximum lags can be
specified, the `lookahead` parameter ensures that additional recursive executions are performed so the first solution
found is not preemptively accepted, the minimum segment length(`minlength`) can also be specified if FWHM are estimable,
and the minimum cross correlation(`mincorr`) for a match can dictate whether peaks were found to align or not.

*Note* This method works best with flat baselines because it repeats last known values when padding aligned spectra.
It is highly efficient, and in my tests does a good job, but other methods definitely exist. Let me know if other peak Alignment
methods are important for your work-flow, I'll see if I can implement them.

Application of Fast Fourier Transform Cross-Correlation for the Alignment of Large Chromatographic and Spectral Datasets
Jason W. H. Wong, Caterina Durante, and, Hugh M. Cartwright. Analytical Chemistry 2005 77 (17), 5655-5661
"""
function RAFFT(raw, reference; maxlags::Int = 500, lookahead::Int = 1, minlength::Int = 20, mincorr::Float64 = 0.05)
    corrected = zeros( size( raw ) )
    for sample in 1 : size( raw )[ 1 ]
        corrected[ sample, : ] = RecursiveAlignment( raw[ sample, : ], reference;
                                                    maxlags = maxlags,
                                                    lookahead = lookahead,
                                                    minlength = minlength,
                                                    mincorr = mincorr)
    end
    return corrected
end



"""
    AssessHealth( X )

Returns a somewhat detailed Dict containing information about the 'health' of a dataset. What is included is the following:
    - PercentMissing: percent of missing entries (includes nothing, inf / nan) in the dataset
    - EmptyColumns: the columns which have only 1 value
    - RankEstimate: An estimate of the rank of X
    - (optional)Duplicates: returns the rows of duplicate observations
"""
function AssessHealth( X; checkduplicates = true )
    (obs, vars) = size( X )
    NumberMissing = sum( map( x -> isa(x, Missing) || isa(x, Nothing) || isnan(x) || isinf(x), X ) )
    PercentMissing = NumberMissing / (obs * vars)
    empties = []
    for c in 1 : vars
        if length(unique(X[:,c])) == 1
            push!(empties, c)
        end
    end
    RankEst = (NumberMissing == 0) ? LinearAlgebra.rank(X) : missing
    InfoDict = Dict("PercentMissing" => PercentMissing,
                    "EmptyColumns" => empties,
                    "RankEstimate" => RankEst)
    if checkduplicates
        if NumberMissing == 0
            dupes = []
            for o1 in 1:obs
                for o2 in (o1 + 1):obs
                    if X[o1,:] == X[o2,:]
                        push!(dupes, o1)
                    end
                end
            end
            InfoDict["Duplicates"] = dupes
        else
            InfoDict["Duplicates"] = missing
        end
    end
    return InfoDict
end
