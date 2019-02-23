using LinearAlgebra

rbinomial(p, size...) = map( x -> (x < p) ? 1 : 0, rand(size...) )
#forceMatrix(a) = (length(size(a)) == 1) ? reshape( a, length(a), 1 ) : a

struct EchoStateNetwork
    Winput
    W
    Woutput
    States
    LastState
    L2
    alpha
    bias
end

function EchoStateNetwork(X, Y, Reservoir = 1000;
                        L2 = 1e-8, alpha = 0.25, SpectralRadius = 1.00, Sparsity = 0.99, Noise = -1.00,
                        bias = TRUE, burnin = 0)
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
