using StatsBase
#Regression Statistics
ME( y, yhat ) = ( 1.0 / size(Y)[1] ) * sum( ( y - yhat ) )
MAE( y, yhat ) = ( 1.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) ) )
MAPE( y, yhat ) = ( 100.0 / size(Y)[1] ) * sum( abs.( ( y - yhat ) / y ) )

SSE( y, yhat )  = sum( (yhat .- y) .^ 2  )
MSE( y, yhat )  = SSE(y, yhat) / size(y)[1]
RMSE( y, yhat ) = sqrt( SSE(y, yhat) / size(y)[1] )
function PercentRMSE( y, yhat )
    mini = reduce(min, yhat)
    maxi = reduce(max, yhat)
    return RMSE(y,yhat) / (maxi - mini)
end

SSTotal( y )     = sum( ( y    .- StatsBase.mean( y ) ) .^ 2 )
SSReg( y, yhat ) = sum( ( yhat .- StatsBase.mean( y ) ) .^ 2 )
SSRes( y, yhat ) = sum( ( y - yhat ) .^ 2 )

RSquare( y, yhat ) = 1.0 - ( SSRes(y, yhat) / SSTotal(y) )
PearsonCorrelationCoefficient(y, yhat) = StatsBase.cov( y, yhat ) / ( StatsBase.std( y ) * StatsBase.std( yhat )  )
