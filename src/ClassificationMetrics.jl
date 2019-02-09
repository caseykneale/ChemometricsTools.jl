
using LinearAlgebra
using Statistics

struct ClassificationLabel
    ToHot::Dict
    ToCold::Dict
    LabelCount::Int
end

function LabelEncoding(HotOrCold)
    IsCold = false
    IsCold = length( size( HotOrCold ) ) == 1
    if !IsCold
        IsCold = size( HotOrCold )[2] == 1
    end
    if IsCold
        Lbls = unique(HotOrCold)
    else
        Lbls = 1:size(HotOrCold)[2]
    end
    return ClassificationLabel(Dict( Lbls .=> collect(1 : length(Lbls) ) ),
                                Dict( collect(1 : length(Lbls)) .=> Lbls ),
                                length(Lbls) )
end


function ColdToHot(Y, Schema::ClassificationLabel)
    lenY = length( Y )
    Output = zeros( lenY, Schema.LabelCount )
    for y in 1 : lenY
        Output[y, Schema.ToHot[ Y[y] ] ] = 1
    end
    return Output
end

function HotToCold(Y, Schema::ClassificationLabel)
    (lenY, Feats) = size( Y )
    @assert Feats == Schema.LabelCount
    Output = zeros( lenY )
    for y in 1 : lenY
        Output[ y ] =  Schema.ToCold[ findfirst( x -> x == 1, Y[ y , : ] ) ]
    end
    return Output
end

#Y and GT are one cold encodings...
function MulticlassStats(Y, GT, schema; Microaverage = true)
    @assert(size(Y)[1] == size(GT)[1])
    ConfusionMatrix = zeros( schema.LabelCount, schema.LabelCount )
    for y in 1 : size(Y)[1]
        ConfusionMatrix[ schema.ToHot[ Y[ y ] ], schema.ToHot[ GT[ y ] ] ] += 1
    end
    TP = repeat( [0], schema.LabelCount ) ; TN = repeat( [0], schema.LabelCount )
    FP = repeat( [0], schema.LabelCount ) ; FN = repeat( [0], schema.LabelCount )
    for c in 1 : schema.LabelCount
        TP[c] = ConfusionMatrix[c,c]
        FP[c] = sum(ConfusionMatrix[:,c]) - TP[c]
        FN[c] = sum(ConfusionMatrix[c,:]) - TP[c]
        TN[c] = sum(ConfusionMatrix) - TP - FP - FN
    end

    if Microaverage
        TP = StatsBase.mean(TP); TN = StatsBase.mean(TN)
        FP = StatsBase.mean(FP); FN = StatsBase.mean(FN)
        Precision = TP / ( TP + FP )
        Recall = TP / ( TP + FN )
        Sensitivity = TP / ( TP + FN )
        Specificity = TN / ( TN + FP )
        Accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
        FMeasure = 2.0 * ( ( Precision * Recall ) / ( Precision + Recall ) )
        FAR = FP / ( FP + TN )
        FNR = FN / ( FN + TP )
        return Dict("ConfusionMatrix" => ConfusionMatrix,
                    "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                    "Sensitivity" => Sensitivity,   "Specificity" => Specificity,
                    "Precision" => Precision,       "Recall" => Recall,
                    "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                    "FAR" => FAR,                   "FNR" => FNR )
    else #Macro Average
        Precision = StatsBase.mean(TP ./ ( TP .+ FP ))
        Recall = StatsBase.mean(TP ./ ( TP .+ FN ))
        Sensitivity = StatsBase.mean(TP ./ ( TP .+ FN ))
        Specificity = StatsBase.mean(TN ./ ( TN .+ FP ))
        Accuracy = StatsBase.mean(( TP .+ TN ) ./ ( TP .+ TN .+ FP .+ FN ))
        FMeasure = StatsBase.mean(2.0 .* ( ( Precision .* Recall ) ./ ( Precision .+ Recall ) ))
        FAR = StatsBase.mean(FP ./ ( FP .+ TN ))
        FNR = StatsBase.mean(FN ./ ( FN .+ TP ))
        return Dict("ConfusionMatrix" => ConfusionMatrix,
                    "TP" => TP, "FP" => FP, "TN" => TN, "FN" => FN,
                    "Sensitivity" => Sensitivity,   "Specificity" => Specificity,
                    "Precision" => Precision,       "Recall" => Recall,
                    "Accuracy" => Accuracy,         "FMeasure" => FMeasure,
                    "FAR" => FAR,                   "FNR" => FNR )
    end
end