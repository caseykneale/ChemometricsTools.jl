function MakeIntervals( columns::Int, intervalsize::Int = 20 )
    ColSize = columns
    intlen = floor(ColSize / intervalsize) |> Int64
    Remainder = ColSize % intervalsize
    Intervals = [ (1 + ((i-1)*intervalsize)):(i*intervalsize) for i in 1:intlen ]
    if Remainder <= (intlen / 2)
        Intervals[end] = Intervals[end][1] : ColSize
    else
        push!(Intervals, last(Intervals[end]) : ColSize)
    end
    return Intervals
end
#Intervals = MakeInterval( 20, 3 );

function MakeIntervals( columns::Int, intervalsizes::Union{Array, Tuple} = [20, 50, 100] )
    Intervals = Dict()
    for interval in intervalsizes
        Intervals[interval] = MakeIntervals(columns,  interval)
    end
    return Intervals
end

# for Interval in MakeIntervals( 20, [3, 5, 10] )
#     println(first(Interval))
# end


#Weights regression outputs by their relative error
#Square error has mathematical gaurantees - so it's default..
function stackedweights(ErrVec; power = 2)
    SqErr = (1.0 ./ ErrVec) .^ power
    return SqErr / sum(SqErr)
end
