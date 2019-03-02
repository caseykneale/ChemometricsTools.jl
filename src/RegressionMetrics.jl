"""
    ME( y, yhat )

Calculates Mean Error from vectors `Y` and `YHat`
"""
function ME( y, yhat ) = ( 1.0 / size(Y)[1] ) * sum( ( y - yhat ) )

"""
    MAE( y, yhat )

Calculates Mean Average Error from vectors `Y` and `YHat`
"""
function MAE( y, yhat ) = ( 1.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) ) )

"""
    MAPE( y, yhat )

Calculates Mean Average Percent Error from vectors `Y` and `YHat`
"""
function MAPE( y, yhat ) = ( 100.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) / y ) )

"""
    SSE( y, yhat )

Calculates Sum of Squared Errors from vectors `Y` and `YHat`
"""
function SSE( y, yhat )  = sum( (yhat .- y) .^ 2  )

"""
    MSE( y, yhat )

Calculates Mean Squared Error from vectors `Y` and `YHat`
"""
function MSE( y, yhat )  = SSE(y, yhat) / size(y)[1]

"""
    RMSE( y, yhat )

Calculates Root Mean Squared Error from vectors `Y` and `YHat`
"""
function RMSE( y, yhat ) = sqrt( SSE(y, yhat) / size(y)[1] )

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
function SSTotal( y ) = sum( ( y    .- StatsBase.mean( y ) ) .^ 2 )

"""
    SSReg( y, yhat )

Calculates Sum of Squared Deviations due to Regression from vectors `Y` and `YHat`
"""
function SSReg( y, yhat ) = sum( ( yhat .- StatsBase.mean( y ) ) .^ 2 )

"""
    SSRes( y, yhat )

Calculates Sum of Squared Residuals from vectors `Y` and `YHat`
"""
function SSRes( y, yhat ) = sum( ( y - yhat ) .^ 2 )

"""
    RSquare( y, yhat )

Calculates R^2 from `Y` and `YHat`
"""
function RSquare( y, yhat ) = 1.0 - ( SSRes(y, yhat) / SSTotal(y) )

"""
    PearsonCorrelationCoefficient( y, yhat )

Calculates The Pearson Correlation Coefficient from vectors `Y` and `YHat`
"""
function PearsonCorrelationCoefficient(y, yhat) = StatsBase.cov( y, yhat ) / ( StatsBase.std( y ) * StatsBase.std( yhat )  )
