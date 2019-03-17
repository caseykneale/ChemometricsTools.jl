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
    MulticlassStats(Y, GT, schema; Microaverage = true)

Calculates many essential classification statistics based on predicted values `Y`, and ground truth values `GT`, using
the encoding `schema`. Returns a dictionary of many statistics...
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

    if Microaverage
        TP = StatsBase.mean(TP); TN = StatsBase.mean(TN)
        FP = StatsBase.mean(FP); FN = StatsBase.mean(FN)
        Precision = TP / ( TP + FP )
        Recall = TP / ( TP + FN )
        Specificity = TN / ( TN + FP )
        Accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
        FMeasure = 2.0 * ( ( Precision * Recall ) / ( Precision + Recall ) )
        FAR = FP / ( FP + TN )
        FNR = FN / ( FN + TP )
        return Dict("ConfusionMatrix" => ConfusionMatrix,
                    "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                    "Specificity" => Specificity,
                    "Precision" => Precision,       "Recall" => Recall,
                    "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                    "FAR" => FAR,                   "FNR" => FNR )
    else #Macro Average
        Precision = StatsBase.mean(TP ./ ( TP .+ FP ))
        Recall = StatsBase.mean(TP ./ ( TP .+ FN ))
        Specificity = StatsBase.mean(TN ./ ( TN .+ FP ))
        Accuracy = StatsBase.mean(( TP .+ TN ) ./ ( TP .+ TN .+ FP .+ FN ))
        FMeasure = StatsBase.mean(2.0 .* ( ( Precision .* Recall ) ./ ( Precision .+ Recall ) ))
        FAR = StatsBase.mean(FP ./ ( FP .+ TN ))
        FNR = StatsBase.mean(FN ./ ( FN .+ TP ))
        return Dict("ConfusionMatrix" => ConfusionMatrix,
                    "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                    "Specificity" => Specificity,
                    "Precision" => Precision,       "Recall" => Recall,
                    "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                    "FAR" => FAR,                   "FNR" => FNR )
    end
end

#Voting Schemes
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
