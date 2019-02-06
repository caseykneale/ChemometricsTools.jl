struct Interval
    intervals::Array
end

function MakeIntervals( X, intervalsize = 10 )
    ColSize = size(X)[2]
    intlen = floor(ColSize / intervalsize)
    Remainder = ColSize % intervalsize
    Intervals = [ (1 + ((i-1)*intervalsize)):(i*intervalsize) for i in 1:intlen ]
    if Remainder <= (intlen / 2)
        Intervals[end] = Intervals[end][1] : ColSize
    else
        push!(Intervals, last(Intervals[end]) : ColSize)
    end
    return Intervals
end
