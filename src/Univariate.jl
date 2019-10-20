abstract type Univariate end

struct UnivariateCalibration <: Univariate
    Slope::Float64
    Offset::Float64
    Slope_Uncertainty::Float64
    Offset_Uncertainty::Float64
    Rsq::Float64
    Fstatistic::Float64
    DOF::Int
end

"""
    UnivariateCalibration( Y, X )

    Performs a univariate least squares regression with a bias/offset term(`y = mx + b`).
Returns a Univariate Calibration object containing uncertainty in parameters, and other statistics.

"""
function UnivariateCalibration( Y, X )
    len = length( X )
    @assert(len == length(Y))
    mu_x, mu_y = sum( X ) / len             , sum( Y ) / len
    SSxx, SSyy = sum( ( X .- mu_x ) .^ 2 )  , sum( ( Y .- mu_y ) .^ 2 )
    SSxy = sum( ( X .- mu_x ) .* ( Y .- mu_y ) )

    m = SSxy / SSxx
    b = mu_y - ( m * mu_x )

    Y_hat = ( m * X ) .+ b
    SSe = sum( (Y .- Y_hat) .^ 2 )
    SSr = SSyy - SSe

    Syx = SSe / ( len - 2 )
    Sm = Syx / SSxx
    Sb = Syx * ( ( 1 / len ) + ( ( mu_x ^ 2 ) / SSxx ) )

    Rsq = SSr / SSyy
    Fstat = ( SSxx - SSe ) / Syx

    return UnivariateCalibration( m, b, Sm, Sb, Rsq, Fstat, len - 2 )
end

(M::UnivariateCalibration)(X) = (X*M.Slope) + M.Offset

struct StandardAddition
    Unknown::Float64
    Slope::Float64
    Offset::Float64
    Unknown_Uncertainty::Float64
    Slope_Uncertainty::Float64
    Offset_Uncertainty::Float64
    Rsq::Float64
    DOF::Int
end

"""
    StandardAddition( Signal, Spike )

    Performs a univariate standard addition calibration. Returns a StandardAddition
object.

"""
function StandardAddition( Signal, Spike )
    len = length( Signal )
    DOF = len - 2
    @assert(len == length(Spike))
    mu_x, mu_y = sum( Spike ) / len             , sum( Signal ) / len
    SSxx, SSyy = sum( ( Spike .- mu_x ) .^ 2 )  , sum( ( Signal .- mu_y ) .^ 2 )
    SSxy = sum( ( Spike .- mu_x ) .* ( Signal .- mu_y ) )

    m = SSxy / SSxx
    #y = mx + b -> y - mx =  b
    b = mu_y - ( m * mu_x )
    Y_hat = ( m * Spike ) .+ b
    #Find X - intercept...
    x0 = -b / m #y = 0 = mx + b => -b / m = x

    Sy = sqrt( sum( (Signal .- Y_hat) .^ 2 ) / DOF )
    Sx = (Sy / abs(m)) * sqrt( (1/len) * ( mu_y^2 / ( m^2 * sum( ( Spike .- mu_x ) .^ 2 )  ) ) )

    SSe = sum( (Signal .- Y_hat) .^ 2 )
    SSr = SSyy - SSe

    Syx = SSe / DOF
    Sm = Syx / SSxx
    Sb = Syx * ( ( 1 / len ) + ( ( mu_x ^ 2 ) / SSxx ) )

    Rsq = SSr / SSyy
    return StandardAddition( x0, m, b, Sy, Sm, Sb, Rsq, DOF )
end

"""
    Confidence_Offset( UC::Univariate; Significance = 0.05 )

Returns a tuple of (-, +) of the estimated offset/bias's confidence interval from a `UC` Univariate type object.

"""
function Confidence_Offset( UC::Univariate; Significance = 0.05 )
    extent = UC.Offset_Uncertainty * quantile( TDist( UC.DOF ), Significance / 2 )
    return ( UC.Offset - extent, UC.Offset + extent )
end

"""
    Confidence_Slope( UC::Univariate; Significance = 0.05 )

Returns a tuple of (-, +) of the estimated slope's confidence interval from a `UC` Univariate type object.

"""
function Confidence_Slope( UC::Univariate; Significance = 0.05 )
    extent = UC.Slope_Uncertainty * quantile( TDist( UC.DOF ), Significance / 2 )
    return ( UC.Slope - extent, UC.Slope + extent )
end
