#This is a workspace for hyperspectral imaging methods
#Maybe some NWAY stuff will fall in here...
#This shouldn't be on master, but I won't put anything here unless I think it'll work.

function ACE(Background, X, Target)
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

#MF is always superior to CEMXiurui Geng,   Luyan Ji, Weitun Yang, Fuxiang Wang, Yongchao Zhao
#https://arxiv.org/pdf/1612.00549.pdf
function MatchedFilter(X, Target)
    if length(size(Background)) == 3
        Background = reshape(Background, prod( size( Background )[1:2 ]), size( Background )[ 3 ]  )
    end
    mu = Statistics.mean(Background, dims = 1)
    mcent = Background .- mu
    covinv = Base.inv( ( 1.0 / size(Background)[1] ) .* (mcent' * mcent) )
    tmu = Target .- mu
    xmu = X .- mu
    numerator = covinv * tmu
    denominator = ( tmu' * covinv * tmu )
    return numerator / denominator
end
