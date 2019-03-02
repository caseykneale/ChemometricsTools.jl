struct QQ
    StoredTuple::Tuple
end

"""
    QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )

Returns a Plot object of a Quantile-Quantile plot between vectors `Y1` and `Y2` at the desired `Quantiles`.
"""
function QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )
    return QQ( (Statistics.quantile!(Y1, Quantiles), Statistics.quantile!(Y2, Quantiles) ))
end

struct BlandAltman
    means::Array{Float64, 1}
    differences::Array{Float64, 1}
    UpperLimit::Float64
    Center::Float64
    LowerLimit::Float64
    Outliers::Array{Float64}
end

"""
    BlandAltman(Y1, Y2; Confidence = 1.96)

Returns a Plot object of a Bland-Altman plot between vectors `Y1` and `Y2` with a confidence limit of `Confidence`.
"""
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


"""
    plotchem(QQ::{QQ, BlandAltman}; title )

returns either a QQ Plot or a Bland-Altman plot with the defined `title`
"""
function plotchem(QQ::QQ; title = "Quantile-Quantile Plot" )
    a = scatter( QQ.StoredTuple, title = title, xlabel = "Quantile 1", ylabel = "Quantile 2",
                    label = "QQ")
    diff(arr) = [ arr[d] - arr[d+1] for d in collect(1 : ( length( arr ) - 1 ))  ]
    m = sum( diff( QQ.StoredTuple[ 1 ] ) ) / sum( diff( QQ.StoredTuple[ 2 ] ) )
    int = QQ.StoredTuple[2][2] - (m * QQ.StoredTuple[1][2])
    Plots.abline!(a, m, int, label = "Trend" )
end

function plotchem(BA::BlandAltman; title = "Bland Altman")
    a = scatter( ( BA.means, BA.differences ), title = title, xlabel = "Means",
                    ylabel = "Differences", label = "B.A.")
    Plots.abline!(a, 0, BA.UpperLimit, color = :red )
    Plots.abline!(a, 0, BA.Center, color = :red )
    Plots.abline!(a, 0, BA.LowerLimit, color = :red )
end


rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

"""
    IntervalOverlay(Spectra, Intervals, Err)

Displays the relative error(`Err`) of each `interval` ontop of a `Spectra`.
"""
function IntervalOverlay(Spectra, Intervals, Err)
    RelativeErr = Err ./ sum(Err);
    a = plot(Spectra', xlabel = "bins", ylabel = "Absorbance / Relative CV Error", ylim = (0,7), legend = false);
    for (i, line) in enumerate( Intervals )
        scalethis = reduce(max, Spectra)  / reduce(max, RelativeErr)
        w = line[end] - line[1]
        h = RelativeErr[i] * scalethis
        xloc = line[1]
        plot!(rectangle(w,h,xloc,0), opacity=.5)
    end
    return a
end
