using Statistics
using StatsBase
using Plots

struct QQ
    StoredTuple::Tuple
end

function QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )
    return QQ( (Statistics.quantile!(Y1, Quantiles), Statistics.quantile!(Y2, Quantiles) ))
end

function plotchem(QQ; title = "Quantile-Quantile Plot" )
    a = scatter( QQ.StoredTuple, title = title, xlabel = "Quantile 1", ylabel = "Quantile 2",
                    label = "QQ")
    diff(arr) = [ arr[d] - arr[d+1] for d in collect(1 : ( length( arr ) - 1 ))  ]
    m = sum( diff( QQ.StoredTuple[ 1 ] ) ) / sum( diff( QQ.StoredTuple[ 2 ] ) )
    int = QQ.StoredTuple[2][2] - (m * QQ.StoredTuple[1][2])
    Plots.abline!(a, m, int, label = "Trend" )
end


struct BlandAltman
    means::Array{Float64, 1}
    differences::Array{Float64, 1}
    UpperLimit::Float64
    Center::Float64
    LowerLimit::Float64
    Outliers::Array{Float64}
end

function BlandAltman(Y1, Y2; Confidence = 1.96)
    means = (Y1 .+ Y2) ./ 2.0
    diffs = Y2 .- Y1
    MeanofDiffs = StatsBase.mean( diffs )
    StdofDiffs = StatsBase.std( diffs )

    UpperLimit = MeanofDiffs + Confidence * StdofDiffs
    Center = MeanofDiffs
    LowerLimit = MeanofDiffs - Confidence * StdofDiffs
    #To:Do Add trend-line....
    Outliers = findall( (diffs .> MeanofDiffs + Confidence*StdofDiffs) )
    Outliers = vcat(Outliers, findall( diffs .< MeanofDiffs - Confidence*StdofDiffs ) )
    return BlandAltman( means, diffs, UpperLimit, Center, LowerLimit, Outliers )
end


function plotchem(BA::BlandAltman; title = "Bland Altman")
    a = scatter( ( BA.means, BA.differences ), title = title, xlabel = "Means",
                    ylabel = "Differences", label = "B.A.")
    Plots.abline!(a, 0, BA.UpperLimit, color = :red )
    Plots.abline!(a, 0, BA.Center, color = :red )
    Plots.abline!(a, 0, BA.LowerLimit, color = :red )
end
