SquareEuclideanDistance(X, Y) = ( sum(X .^ 2, dims = 2) .+ sum(Y .^ 2, dims = 2)') .- (2 * X * Y')

EuclideanDistance(X, Y) = sqrt.(SquareEuclideanDistance(X, Y))

#Only computes half distance and mirrors it...
function ManhattanDistance(X)
    Result = zeros(size(X)[1], size(X)[1])
    for rowx in 2 : (size(X)[1]), rowy in 1 : (rowx-1)
        Result[rowx, rowy] = sum( abs.( X[rowx,:] - X[rowy,:]  ) )
        Result[rowy, rowx] = Result[rowx, rowy]
    end
    return Result
end

function ManhattanDistance(X, Y)
    Result = zeros(size(X)[1], size(Y)[1])
    for rowx in 1 : (size(X)[1]), rowy in 1 : size(Y)[1]
        Result[rowx, rowy] = sum( abs.( X[rowx,:] - Y[rowy,:]  ) )
    end
    return Result
end
