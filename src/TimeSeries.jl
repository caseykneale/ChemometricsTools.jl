struct RollingWindow
    WindowSize::Int
    observations::Int
    maxiter::Int
    skip::Int
end

"""
    RollingWindow(samples::Int,windowsize::Int)

Creates a RollingWindow iterator from a number of `samples` and a static `windowsize`. The iterator can be used
in for loops to iteratively return indices of a dynamic rolling window.
"""
RollingWindow(samples::Int,windowsize::Int) = RollingWindow(windowsize, samples,samples - windowsize + 1, 1)

"""
    RollingWindow(samples::Int,windowsize::Int,skip::Int)

Creates a RollingWindow iterator from a number of `samples` and a static `windowsize` where every iteration `skip` steps are skipped.
The iterator can be used in for loops to iteratively return indices of a dynamic rolling window.
"""
RollingWindow(samples::Int,windowsize::Int,skip::Int) = RollingWindow(windowsize, samples,samples - windowsize + 1, skip)

function Base.iterate( iter::RollingWindow, state = 1 )
    window = 0
    if state <= iter.maxiter
        window = collect( state : ( state + iter.WindowSize - 1) )
    else
        return nothing
    end
    return ( window ,  state + iter.skip  )
end

mutable struct ewma
    lambda::Float64
    center::Float64
    lastval::Float64
    rv::RunningVar
end

"""
    EWMA(Initial::Float64, Lambda::Float64) = ewma(Lambda, Initial, Initial, RunningVar(Initial))

Constructs an exponentially weighted moving average object from an initial scalar property value `Initial` and
the decay parameter `Lambda`. This defaults the center value to be the initial value.
"""
EWMA(Initial::Float64, Lambda::Float64) = ewma(Lambda, Initial, Initial, RunningVar(Initial))
"""
    EWMA(Initial::Float64, Lambda::Float64) = ewma(Lambda, Initial, Initial, RunningVar(Initial))

Constructs an exponentially weighted moving average object from an vector of scalar property values `Initial` and
the decay parameter `Lambda`. This computes the running statistcs neccesary for creating the EWMA model using the
interval provided and updates the center value to the mean of the provided values.
"""
function EWMA(Initial::Array, Lambda::Float64)
    burnin = EWMA(Initial[1], Lambda)#Call constructor above
    len = length(Initial)
    for sample in 1:len
        burnin(Initial[sample])
    end
    burnin.center = burnin.rv.m.mu
    return burnin
end
"""
    EWMA(P::ewma)(New; train = true)

Provides an EWMA score for a `New` scalar value. If ```train == true``` the model is updated to include this new value.
"""
function (P::ewma)(New; train = true)
    P.lastval = ( P.lambda * New ) + ( 1 - P.lambda ) * P.lastval
    if train == true
        Update!(P.rv, New)
    end
    return P.lastval
end
"""
    ChangeCenter(P::ewma, new::Float64)

This is a convenience function to update the center of a `P` EWMA model, to a `new` scalar value.
"""
function ChangeCenter(P::ewma, new::Float64)
    P.center .= new
end
"""
    Variance(P::ewma)

This function returns the EWMA control variance.
"""
Variance(P::ewma) = (P.lambda / (2.0 - P.lambda) ) * Variance(P.rv)
"""
    Limits(P::ewma; k = 3.0)

This function returns the upper and lower control limits with a `k` span of variance for an EWMA object `P`.
"""
Limits(P::ewma; k = 3.0) = (P.center + (k * sqrt( Variance( P ) ) ), P.center - (k * sqrt( Variance( P ) ) )  )


mutable struct NaiveForecaster
    lastval::Float64
end
"""
    NaiveForecaster( univariate::Array )

Makes a forecaster model that simply predicts the last value it has seen for all future values.
"""
NaiveForecast(univariate::Array) = NaiveForecaster( last( univariate ) )
"""
    update!( model::NaiveForecaster, newdata::Float64 )

Makes a forecast model that simply predicts the last value it has seen for all future values.
"""
update!( model::NaiveForecaster, newdata::Float64 ) = model.lastval .= newdata

"""
    update( model::NaiveForecaster, newdata::Float64 )

returns an updated forecast model that simply predicts the last value it has seen for all future values.
"""
update(model::NaiveForecaster, newdata::Float64) = return NaiveForecaster( newdata )

"""
    (nf::NaiveForecaster)()

Predicts with a naive forecaster model.
"""
(nf::NaiveForecaster)() = nf.lastval

mutable struct SimpleAverage
    runmean::RunningMean
end
"""
    SimpleAverage( univariate )
"""
SimpleAverage(initial::Float64) = SimpleAverage( RunningMean( initial) )

"""
    SimpleAverage( univariate::Array )
"""
function SimpleAverage(univariate::Array)
    rm = RunningMean( univariate[1])
    for i in 2:length(univariate)
        Update!( rm, univariate[ i ] )
    end
    return SimpleAverage( rm )
end
"""
    Update( model::naiveforecaster, newdata::Float64 )
"""
function Update(model::SimpleAverage, newdata::Float64)
    return Update!( model.runmean, newdata )
end

"""
    (sa::SimpleAverage)()

Predicts with a naive forecaster model.
"""
(sa::SimpleAverage)() = sa.runmean.mu


struct EchoStateNetwork
    Winput::Array
    W::Array
    Woutput::Array
    States::Array
    LastState::Array
    L2::Float64
    alpha::Float64
    bias
end

"""
    EchoStateNetwork(X, Y, Reservoir = 1000;
                        L2 = 1e-8, alpha = 0.25, SpectralRadius = 1.00, Sparsity = 0.99, Noise = -1.00,
                        bias = true, burnin = 0)

    Currently untested.
"""
function EchoStateNetwork(X, Y, Reservoir = 1000;
                        L2 = 1e-8, alpha = 0.25, SpectralRadius = 1.00, Sparsity = 0.99, Noise = -1.00,
                        bias = true, burnin = 0)
    X = forceMatrix( X ) ; Y = forceMatrix( Y )#Ensure inputs are Array2's
    (Obs, Vars) = size( X )
    #Allocate memory for the collected states matrix
    States = zeros(bias + Vars + Reservoir, Obs - burnin - 1)
    Predictions = zeros( size( Y ) )
    #Initialize input weight matrix and recurrent weight matrix
    Winput = rand( Reservoir, bias + Vars) .* -2 .+ 1
    #Generate Resevoir matrix with random uniform values from -1 to +1
    W = rand( Reservoir, Reservoir ) .* -2 .+ 1
    #Create a 'mask' from the Bernouli distribution (binomial distribution size = 1)
    Dropout = rbinomial(1.0 - Sparsity, Reservoir, Reservoir )
    #Now we need to introduce sparsity in our reservoir connections based on the mask
    W = W .* Dropout
    #Normalize and set spectral radius of reservoir weight matrix
    W = W .* ( SpectralRadius / abs( LinearAlgebra.eigen( W ).values[ 1 ] ) )
    #Run the reservoir with the data and collect X
    state = zeros(Reservoir, 1)
    #Go through each time step and compute the model
    for t in 1 : (Obs - 1)
        u = bias ? vcat( 1, X[t,:] ) : X[t,:]
        state = ((1.0 - alpha) * state) .+ (alpha * tanh.( Winput * u .+ W * state ))
        #If we are past the burnin number, start saving the model
        if t > burnin
            States[ :, t - burnin ] = vcat(u, state)
        end
    end
    if Noise > 0.0
        VS = size(S)[2]
        for obs in 1:Obs
            mini = reduce( min, States[obs,:] )
            maxi = reduce( max, States[obs,:] )
            States[obs,:] .+= States[obs,:] .* ( rand( 1, VS ) .* Noise .- (Noise/2.0) )
        end
    end
    #Train the output layer from the random reservoir via least squares
    Woutput = Y[ (burnin + 2) : end, :]' * States' * Base.inv( States * States' .+ L2 * LinearAlgebra.Diagonal(ones(bias + Vars + Reservoir) ) )
    return EchoStateNetwork( Winput, W, Woutput, States, state, L2, alpha, bias )
end

#This function modifies an ESN model's output weights without reconstructing the dynamic resevoir.
#Time saver!
function TuneRidge(Y, model::EchoStateNetwork, L2, burnin = 0 )
    model.L2 .= L2
    model.Woutput .= Y[ (burnin + 2) : end]' * model.States' * Base.inv( model.States * model.States' .+ L2 * diag(ones(bias + Vars + Reservoir) ) )
end

#This function uses X data to predict Y values from a stored ESN model.
#Burnin can be used here, but it simply rejects the first N samples from storage/allocation.
#UseLastState will employ the last known state from the training set as the first state estimate (reccomended)
function PredictFn(model::EchoStateNetwork, X; UseLastState = true, burnin = 0)
    (Obs,Vars) = size(X)
    #Run the trained ESN using last state as starting point
    state = UseLastState ? model.LastState : state = zeros( size( model.Reservoir ) )
    Prediction = zeros( size(model.Woutput)[1], Obs - burnin)

    u = X[1,:]
    for t in 1 : Obs
        u = model.bias ? vcat( 1, X[t,:]) : X[t,:]
        state = ((1.0 - model.alpha ) * state) .+ (model.alpha * tanh.( model.Winput * u .+ model.W * state ) )
        if t > burnin
            Prediction[ : , t - burnin ] = model.Woutput * vcat(u, state)
        end
        if t < Obs
            u = X[t + 1,:] #Load next sample for prediction
        end
    end#End for

    return Prediction
end
