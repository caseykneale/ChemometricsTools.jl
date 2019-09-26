"""
"""
@recipe function residualsplotrecipe(rmodel::ChemometricsTools.RegressionModels, X, Y)
    Yhat = rmodel(X)
    seriestype := :scatter
    title := "Residual vs Fitted Values Plot"
    xlabel := Symbol("Fitted Values")
    ylabel := Symbol("Residuals")
    @series y := (Yhat .- Y, Yhat)
end

struct QQ
    StoredTuple::Tuple
end

"""
    QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )

Returns a plotable object of a Quantile-Quantile plot between vectors `Y1` and `Y2` at the desired `Quantiles`.
"""
function QQ( Y1, Y2; Quantiles = collect( 1 : 99 ) ./ 100 )
    return QQ( (Statistics.quantile!(Y1, Quantiles), Statistics.quantile!(Y2, Quantiles) ))
end

@recipe function f(qq::QQ)
    m = sum( diff( qq.StoredTuple[ 1 ] ) ) / sum( diff( qq.StoredTuple[ 2 ] ) )
    int = qq.StoredTuple[2][2] - (m * qq.StoredTuple[1][2])

    seriestype := :scatter
    title := "Quantile-Quantile Plot"
    xlabel := Symbol("Quantile 1")
    ylabel := Symbol("Quantile 2")
    @series y := qq.StoredTuple

    seriestype := :straightline
    label := "Trend-Line"
    @series y := ([0, 1], [int, int + m])
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

Returns a Plottable object of a Bland-Altman plot between vectors `Y1` and `Y2` with a confidence limit of `Confidence`.
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

@recipe function f(BA::BlandAltman)
    seriestype := :scatter
    title := "Bland Altman"
    xlabel := Symbol("Means")
    ylabel := Symbol("Differences")
    @series y := ( BA.means, BA.differences )
    seriestype := :straightline
    color := :red
    @series y := ([0, 1], [BA.UpperLimit, BA.UpperLimit])
    seriestype := :straightline
    color := :red
    @series y := ([0, 1], [BA.Center, BA.Center])
    seriestype := :straightline
    color := :red
    @series y := ([0, 1], [BA.LowerLimit, BA.LowerLimit])
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
"""
    IntervalOverlay(Spectra, Intervals, Err)

Creates a Plottable object to display the relative error(`Err`) of each `interval` ontop of a `Spectra`.
"""
struct IntervalOverlay
    Spectra
    Intervals
    Err
end

@recipe function f(IO::IntervalOverlay)
    RelativeErr = IO.Err ./ sum(IO.Err);

    seriestype := :path
    title := "Spectral Error Overlay"
    xlabel := Symbol("Bins")
    ylabel := Symbol("Absorbance / Relative CV Error")
    legend := false
    @series y := IO.Spectra'

    seriestype := :shape
    opacity := 0.5
    for (i, line) in enumerate( IO.Intervals )
        scalethis = reduce(max, IO.Spectra)  / reduce(max, RelativeErr)
        w = line[end] - line[1]
        h = RelativeErr[i] * scalethis
        xloc = line[1]
        @series y := rectangle(w,h,xloc,0)
    end
end

"""
    PCA/LDA(::Union{PCA, LDA}; Axis = [1,2])

Plots scores of PCA/LDA analysis.
"""
@recipe function f(DA::Union{PCA, LDA}; Axis = [1,2])
    @assert(length(Axis) == 2, "Axis for PCA/LDA plot must be of length 2." )
    exvar = round.(ExplainedVariance(DA) .* 10000) /100
    seriestype := :scatter
    if isa(DA, PCA)
        title := "PCA Scores Plot"
        xlabel := "PCA $(Axis[1]) [$(exvar[Axis[1]]) %]"
        ylabel := "PCA $(Axis[2]) [$(exvar[Axis[2]]) %]"
    else
        title := "LDA Scores Plot"
        xlabel := "DA $(Axis[1]) [$(exvar[Axis[1]]) %]"
        ylabel := "DA $(Axis[2]) [$(exvar[Axis[2]]) %]"
    end
    @series y := ( DA.Scores[:,Axis[2]], DA.Scores[:,Axis[1]] )
end

# x = -pi:(pi/200):pi;
# A = transpose((sin.(x) .+ 1.0) .+ randn(400,3) ./ 50);
# Intervals = [ (((i-1) * 10) + 1, ((i) * 10)) for i in 1:40 ];
# Err = rand(40);
# IOo = IntervalOverlay(A,Intervals,Err)
# Plots.plot(IOo)

"""
    DiscriminantAnalysisPlot(DA, GD, YHot, LblEncoding, UnlabeledData, Axis = [1,2], Confidence = 0.90)
...
"""
struct DAPlot
    DA
    GD
    YHot
    LblEncoding
    UnlabeledData
    Axis
    Confidence
end

DiscriminantAnalysisPlot(DA, GD, YHot, LblEncoding, UnlabeledData;
                    Axis = [1,2], Confidence = 0.90) = DAPlot( DA, GD, YHot,
                    LblEncoding, UnlabeledData, Axis, Confidence)

@recipe function f(dap::DAPlot)
    exvar = round.(ExplainedVariance(dap.DA) .* 10000) /100
    @assert all(dap.Axis .<= size(dap.UnlabeledData)[2])
    A = []
    Classes = size(dap.YHot)[2]
    colors = get_color_palette(:auto, plot_color(:white), Classes)
    for c in 1:Classes
        #Labelled data
        whichrows = dap.YHot[:,c] .== 1.0
        seriestype := :scatter
        color := colors[c]
        label := dap.LblEncoding.ToCold[c]
        @series y := ( dap.DA.Scores[whichrows,dap.Axis[1]], dap.DA.Scores[whichrows,dap.Axis[2]] )
        #Make confidence ellipse
        elipse = ConfidenceEllipse(dap.GD.ProjectedClassCovariances[c], dap.GD.ProjectedClassMeans[c,:],
                    dap.Confidence, dap.Axis; pointestimate = 180 );

        seriestype := :path
        color := :black
        linestyle := :dash
        linewidth := 2
        label := ""
        @series y := (elipse[:,1], elipse[:,2])
    end
    #New Data
    seriestype := :scatter
    color := :black
    markershape = :star
    label := ""
    legend := :topleft
    xlabel := "DA $(dap.Axis[1]) [$(exvar[dap.Axis[1]]) %]"
    ylabel := "DA $(dap.Axis[2]) [$(exvar[dap.Axis[2]]) %]"
    @series y := ( dap.UnlabeledData[:,dap.Axis[1]] , dap.UnlabeledData[:,dap.Axis[2]] )
end
