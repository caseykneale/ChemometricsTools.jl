SquareEuclideanDistance(X) = ( sum(X .^ 2, dims = 2) .+ sum(X .^ 2, dims = 2)') .- (2 * X * X')
EuclideanDistance(X) = sqrt.(SquareEuclideanDistance(X))

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

#Kernels
struct Kernel
    params::Union{Float64, Tuple}
    ktype::String
    original::Array
end
Kernel( X ) = Kernel(0.0, "linear", X)

#This is just a wrapper so we can apply kernels willy nilly in one line
function (K::Kernel)(X)
    if K.ktype == "linear"
        return LinearKernel(K.original, X, K.params)
    elseif K.ktype == "gaussian" || K.ktype == "rbf"
        return GaussianKernel(K.original , X, K.params)
    end
end

LinearKernel(X, c) =  (X * X') .+ c
LinearKernel(X, Y, c) = (Y * X') .+ c

function GaussianKernel(X, sigma)
    Gamma = -1.0 / (2.0 * sigma^2)
    return exp.( SquareEuclideanDistance(X) .* Gamma  )
end

function GaussianKernel(X, Y, sigma)
    Gamma = -1.0 / (2.0 * sigma^2)
    return exp.( SquareEuclideanDistance(Y, X) .* Gamma  )
end
