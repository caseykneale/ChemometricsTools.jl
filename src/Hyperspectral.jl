#This is a workspace for hyperspectral imaging methods

"""
    ACE(Background, X, Target)

Untested
"""
function ACE(Background, X, Target)
    @assert( length(size(Background)) < 4 )
    if length(size(Background)) == 3
        Background = reshape(Background, prod( size( Background )[1:2 ]), size( Background )[ 3 ]  )
    end
    mu = Statistics.mean(Background, dims = 1)
    mcent = Background .- mu
    covinv = Base.inv( ( 1.0 / size(Background)[1] ) .* (mcent' * mcent) )
    tmu = Target .- mu
    xmu = X .- mu
    numerator = tmu' * covinv * xmu
    denominator = ( tmu' * covinv * tmu ) * ( xmu' * covinv * xmu )
    return (numerator * numerator) / denominator
end

"""
    MatchedFilter(X, Target)

Untested

MatchedFilter is always superior to CEM. Xiurui Geng, Luyan Ji, Weitun Yang, Fuxiang Wang, Yongchao Zhao
https://arxiv.org/pdf/1612.00549.pdf
"""
function MatchedFilter(X, Target)
    @assert( length(size(X)) < 4 )
    if length(size(X)) == 3
        X = reshape(X, prod( size( X )[1:2 ]), size( X )[ 3 ]  )
    end
    mu = Statistics.mean(X, dims = 1)
    mcent = X .- mu
    covinv = Base.inv( ( 1.0 / size(X)[1] ) .* (mcent' * mcent) )
    tmu = Target .- mu
    xmu = X .- mu
    numerator = covinv * tmu
    denominator = ( tmu' * covinv * tmu )
    return numerator / denominator
end
