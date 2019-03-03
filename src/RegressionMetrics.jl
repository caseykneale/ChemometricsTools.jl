"""
    ME( y, yhat )

Calculates Mean Error from vectors `Y` and `YHat`.
"""
function ME( y, yhat )
    return ( 1.0 / size(Y)[1] ) * sum( ( y - yhat ) )
end

"""
    MAE( y, yhat )

Calculates Mean Average Error from vectors `Y` and `YHat`
"""
function MAE( y, yhat )
    return ( 1.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) ) )
end

"""
    MAPE( y, yhat )

Calculates Mean Average Percent Error from vectors `Y` and `YHat`
"""
function MAPE( y, yhat )
    return ( 100.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) / y ) )
end

"""
    SSE( y, yhat )

Calculates Sum of Squared Errors from vectors `Y` and `YHat`
"""
function SSE( y, yhat )
    return sum( (yhat .- y) .^ 2  )
end

"""
    MSE( y, yhat )

Calculates Mean Squared Error from vectors `Y` and `YHat`
"""
function MSE( y, yhat )
    return SSE(y, yhat) / size(y)[1]
end

"""
    RMSE( y, yhat )

Calculates Root Mean Squared Error from vectors `Y` and `YHat`
"""
function RMSE( y, yhat )
    return sqrt( SSE(y, yhat) / size(y)[1] )
end

"""
    PercentRMSE( y, yhat )

Calculates Percent Root Mean Squared Error from vectors `Y` and `YHat`
"""
function PercentRMSE( y, yhat )
    mini = reduce(min, yhat)
    maxi = reduce(max, yhat)
    return RMSE(y,yhat) / (maxi - mini)
end

"""
    SSTotal( y, yhat )

Calculates Total Sum of Squared Deviations from vectors `Y` and `YHat`
"""
function SSTotal( y )
    return sum( ( y .- StatsBase.mean( y ) ) .^ 2 )
end

"""
    SSReg( y, yhat )

Calculates Sum of Squared Deviations due to Regression from vectors `Y` and `YHat`
"""
function SSReg( y, yhat )
    return sum( ( yhat .- StatsBase.mean( y ) ) .^ 2 )
end

"""
    SSRes( y, yhat )

Calculates Sum of Squared Residuals from vectors `Y` and `YHat`
"""
function SSRes( y, yhat )
    return sum( ( y - yhat ) .^ 2 )
end
"""
    RSquare( y, yhat )

Calculates R^2 from `Y` and `YHat`
"""
function RSquare( y, yhat )
    return 1.0 - ( SSRes(y, yhat) / SSTotal(y) )
end
"""
    PearsonCorrelationCoefficient( y, yhat )

Calculates The Pearson Correlation Coefficient from vectors `Y` and `YHat`
"""
function PearsonCorrelationCoefficient(y, yhat)
    return StatsBase.cov( y, yhat ) / ( StatsBase.std( y ) * StatsBase.std( yhat )  )
end
