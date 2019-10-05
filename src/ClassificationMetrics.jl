
"""
    IsColdEncoded(Y)

Returns a boolean true if the array Y is cold encoded, and false if not.
"""
IsColdEncoded(Y) = size( forceMatrix( Y ) )[2] == 1

struct ClassificationLabel
    ToHot::Dict
    ToCold::Dict
    LabelCount::Int
end

""""
    LabelEncoding(HotOrCold)

Determines if an Array, `Y`, is one hot encoded, or cold encoded by it's dimensions.
Returns a ClassificationLabel object/schema to convert between the formats.
"""
function LabelEncoding(HotOrCold)
    HotOrCold = forceMatrix(HotOrCold)
    if IsColdEncoded(HotOrCold)
        Lbls = unique(HotOrCold)
    else
        Lbls = (1:size(HotOrCold)[2])
    end
    return ClassificationLabel(Dict( Lbls .=> collect(1 : length(Lbls) ) ),
                                Dict( collect(1 : length(Lbls)) .=> Lbls ),
                                length(Lbls) )
end

"""
    ColdToHot(Y, Schema::ClassificationLabel)

Turns a cold encoded `Y` vector into a one hot encoded array.
"""
function ColdToHot(Y, Schema::ClassificationLabel)
    lenY = length( Y )
    Output = zeros( lenY, Schema.LabelCount )
    for y in 1 : lenY
        Output[y, Schema.ToHot[ Y[y] ] ] = 1
    end
    return Output
end

"""
    HotToCold(Y, Schema::ClassificationLabel)

Turns a one hot encoded `Y` array into a cold encoded vector.
"""
function HotToCold(Y, Schema::ClassificationLabel)
    Y = forceMatrix(Y)
    (lenY, Feats) = size( Y )
    @assert Feats == Schema.LabelCount
    Output = Array{Any,2}(undef, lenY, 1 )
    for y in 1 : lenY
        Output[ y ] =  Schema.ToCold[ findfirst( x -> x == 1, Y[ y , : ] ) ]
    end
    return Output
end


"""
    StatsFromTFPN(TP, TN, FP, FN)

Calculates many essential classification statistics based on the numbers of True Positive(`TP`), True Negative(`TN`),
 False Positive(`FP`), and False Negative(`FN`) examples.
"""
function StatsFromTFPN(TP, TN, FP, FN)
    ConfusionMatrix = reshape( [TP, FP, FN, TN], 2, 2 )
    Precision = TP / ( TP + FP )
    Recall = TP / ( TP + FN )
    Specificity = TN / ( TN + FP )
    Accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
    FMeasure = 2.0 * ( ( Precision * Recall ) / ( Precision + Recall ) )
    FAR = FP / ( FP + TN )
    FNR = FN / ( FN + TP )
    Prevalence = TP + FN
    return Dict("ConfusionMatrix" => ConfusionMatrix,
                "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                "Specificity" => Specificity,
                "Precision" => Precision,       "Recall" => Recall,
                "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                "FAR" => FAR,                   "FNR" => FNR,
                "Prevalence" => Prevalence )
end

"""
    MulticlassStats(Y, GT, schema; Microaverage = true)

Calculates many essential classification statistics based on predicted values `Y`, and ground truth values `GT`, using
the encoding `schema`. Returns a tuple whose first entry is a dictionary of averaged statistics, and whose second
entry is a dictionary of the form "Class" => Statistics Dictionary ...
"""
function MulticlassStats(Y, GT, schema; Microaverage = true)
    Y = forceMatrix(Y)
    GT = forceMatrix(GT)
    Y = IsColdEncoded(Y) ?  map(x -> Int(schema.ToHot[ x ]), Y) : HighestVote(Y)
    GT = IsColdEncoded(GT) ? map(x -> Int(schema.ToHot[ x ]), GT) : HighestVote(GT)
    @assert(size(Y)[1] == size(GT)[1])
    ConfusionMatrix = zeros( schema.LabelCount, schema.LabelCount )
    for y in 1 : size(Y)[1]
        ConfusionMatrix[ Y[ y ], GT[ y ] ] += 1
    end
    TP = repeat( [0], schema.LabelCount ) ; TN = repeat( [0], schema.LabelCount )
    FP = repeat( [0], schema.LabelCount ) ; FN = repeat( [0], schema.LabelCount )
    for c in 1 : schema.LabelCount
        TP[c] = ConfusionMatrix[c,c]
        FP[c] = sum(ConfusionMatrix[:,c]) - TP[c]
        FN[c] = sum(ConfusionMatrix[c,:]) - TP[c]
        TN[c] = sum(ConfusionMatrix) - TP[c] - FP[c] - FN[c]
    end

    #Compute Class-wise statistics.
    #Make a dictionary which contains for each class, a dictionary of statistics.
    ClasswiseStats = Dict()
    for class in 1 : schema.LabelCount
        ClasswiseStats[schema.ToCold[class]] = StatsFromTFPN(TP[class], TN[class], FP[class], FN[class])
    end
    GlobalStats = Dict()
    if Microaverage
         TP = StatsBase.mean( TP ); TN = StatsBase.mean( TN )
         FP = StatsBase.mean( FP ); FN = StatsBase.mean( FN )
         GlobalStats = Dict(:Global => StatsFromTFPN( TP, TN, FP, FN ) )
    else #Macro Average
        Precision = StatsBase.mean( TP ./ ( TP .+ FP ) )
        Recall = StatsBase.mean( TP ./ ( TP .+ FN ) )
        Specificity = StatsBase.mean( TN ./ ( TN .+ FP ) )
        Accuracy = StatsBase.mean( ( TP .+ TN ) ./ ( TP .+ TN .+ FP .+ FN ) )
        FMeasure = StatsBase.mean( 2.0 .* ( ( Precision .* Recall ) ./ ( Precision .+ Recall ) ) )
        FAR = StatsBase.mean( FP ./ ( FP .+ TN ) )
        FNR = StatsBase.mean( FN ./ ( FN .+ TP ) )
        GlobalStats = Dict(:Global => Dict( "ConfusionMatrix" => ConfusionMatrix,
                            "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                            "Specificity" => Specificity,
                            "Precision" => Precision,       "Recall" => Recall,
                            "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                            "FAR" => FAR,                   "FNR" => FNR ) )
    end
    return (GlobalStats, ClasswiseStats)
end


"""
    StatsDictToDataFrame(DictOfStats, schema)

Converts a dictionary of statistics which is returned from `MulticlassStats` into a labelled dataframe.
This is an intermediate step for automated report generation.
"""
function StatsDictToDataFrame(ClasswiseStats; digits = 4,
                                StatsList = [   "FMeasure", "Accuracy", "Specificity",
                                                "Precision", "Recall", "FAR", "FNR" ])
    ClassName = repeat( [ "No Name" ], length(ClasswiseStats) );
    daf = DataFrame( :Statistics => StatsList)
    for (j, ( classnames, stats )) in enumerate(ClasswiseStats)
        if typeof(classnames) == Symbol
            classnames = string(classnames)
        end
        ClassName[ j ] = classnames
        ClassStats = zeros( length(StatsList) );
        for (i, stat) in enumerate( StatsList )
            ClassStats[ i ] = round( stats[ stat ]; sigdigits = digits )
        end
        daf[ Symbol( classnames ) ] = ClassStats
    end
    return daf
end

"""
    StatsToDataFrame(stats, schema, filepath, name)

Converts the 2-Tuple returned from `MulticlassStats()` (`stats`) to a CSV file with a specified `name`
in a specified `filepath` using the prescribed encoding `schema`.

The statistics associated with the global analysis will end in a file name  of "-global.csv"
and the local statistics for each class will end in a file named "-classwise.csv"
"""
function StatsToCSVs(Stats, filepath, name)
    globaldf = StatsDictToDataFrame(Stats[1])
    localdf = StatsDictToDataFrame(Stats[2])
    CSV.write(Base.joinpath(filepath, name * "-global.csv"), globaldf)
    CSV.write(Base.joinpath(filepath, name * "-classwise.csv"), localdf)
end

"""
    DataFrameToLaTeX( df, caption = "" )

Converts a DataFrame object to a LaTeX table (string).
"""
function DataFrameToLaTeX( df; caption = "")
    (Rows, Columns) = size(df)
    ColmFormat = reduce(*, ["c" for i in 1:Columns])
    ColmNames = join(string.(names(df))," & ")
    retstr = "\\begin{table}[h] \n" *
             "\t\\begin{tabular}{$ColmFormat} \n" *
             "\t\t $ColmNames \\\\ \\hline \n"

    TableInterior = [ "\t\t " * join(string.(values(df[row,:])), " & ") * " \\\\ \n" for row in 1:Rows]
    TableInterior = reduce(*, TableInterior)

    retstr *= TableInterior *
              "\t \\end{tabular} \n"
    retstr *= (length(caption) > 0) ? "\t \\caption{$caption} \n" : ""
    retstr *= "\\end{table}\n"
    return retstr
end

"""
    StatsToLaTeX(Stats, filepath = nothing, name = nothing,
                        digits = 3, maxcolumns = 6; Comment = "",
                        StatsList = [   "FMeasure", "Accuracy", "Specificity",
                                        "Precision", "Recall", "FAR", "FNR" ])

Converts a MulticlassStats object to a LaTeX table (string or saved file).
LaTeX tables contain rows of StatsList, and a maximum column number of maxcolumns.
Information is presented with a set number of decimals(digits).

"""
function StatsToLaTeX(Stats, filepath = nothing, name = nothing;
                        digits = 3, maxcolumns = 6, Comment = "",
                        StatsList = [   "FMeasure", "Accuracy", "Specificity",
                                        "Precision", "Recall", "FAR", "FNR"     ] )
    globaldf = StatsDictToDataFrame(Stats[1]; digits = digits, StatsList = StatsList)
    classes = [ k for k in keys( Stats[2] ) ]
    localdf = []
    if length(classes) > maxcolumns
        maxtbls = Int( round( length( classes ) / maxcolumns ) )
        tmpdf = StatsDictToDataFrame(Stats[2]; digits = digits, StatsList = StatsList)
        for rn in 1 : maxtbls
            colview = []
            if rn < maxtbls
                colview = classes[ ((maxcolumns * (rn - 1)) + 1) : (rn * maxcolumns)]
            else
                colview = classes[ ((maxcolumns * (rn - 1)) + 1) : end]
            end
            push!(localdf, tmpdf[:, Symbol.( vcat(["Statistics"], colview))] )
        end
    else
        localdf = StatsDictToDataFrame(Stats[2]; digits = digits, StatsList = StatsList)
    end
    TimeStamp = Dates.format(now(), "mm-dd-YYYY HH:MM")
    ReportStr = "\\documentclass[]{report}\n" *
                "% Report Generated from ChemometricsTools.jl ($TimeStamp)\n" *
                "% $Comment" *
                "\n\\begin{document}\n"
    ReportStr *= DataFrameToLaTeX( globaldf; caption = "Global Classification Statistics." )
    if length(classes) > maxcolumns
        for tdf in localdf
            ReportStr *= DataFrameToLaTeX( tdf; caption = "Classwise Classification Statistics." )
        end
    else
        ReportStr *= DataFrameToLaTeX( localdf; caption = "Classwise Classification Statistics." )
    end
    ReportStr *= "\n\\end{document}"
    #If no name is given -> Return the string
    if isa(filepath, Nothing) or isa(name, Nothing)
        return ReportStr
    else
        open( filepath * name * ".tex", "w" ) do f
            write( f, ReportStr )
        end
        return true
    end
end

"""
    Threshold(yhat; level = 0.5)

For a binary vector `yhat` this decides if the label is a 0 or a 1 based on it's value relative to a threshold `level`.
"""
Threshold(yhat; level = 0.5) = map( y -> (y >= level) ? 1 : 0, yhat)

"""
    MulticlassThreshold(yhat; level = 0.5)

Effectively does the same thing as Threshold() but per-row across columns.

*Warning this function can allow for no class assignments. HighestVote is preferred*
"""
function MulticlassThreshold(yhat; level = 0.5)
    newY = zeros(size(yhat))
    for obs in 1 : size(yhat)[1]
        (val, ind) = findmax( yhat[obs,:] )
        if val > level
            newY[ind] = val
        end
    end
    return newY
end

"""
    HighestVote(yhat)

Returns the column index for each row that has the highest value in one hot encoded `yhat`. Returns a one cold encoded vector.
"""
function HighestVote( yhat )
    return [ findmax( yhat[obs,:] )[2] for obs in 1 : size(yhat)[1]  ]
end

"""
    HighestVoteOneHot(yhat)

Turns the highest column-wise value to a 1 and the others to zeros per row in a one hot encoded `yhat`. Returns a one cold encoded vector.
"""
function HighestVoteOneHot( yhat )
    (Obs, Classes) = size( yhat )
    ret = zeros( ( Obs, Classes ) )
    for o in 1 : Obs
        ret[ o, argmax( yhat[ o, : ] ) ] = 1
    end
    return ret
end
