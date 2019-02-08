using LinearAlgebra
using DSP


function FirstDerivative(X::Array)
    Xsize = (length(size( X )) > 1) ? size( X ) : (1,length(X))
    if Xsize[1] == 1 #Sorry but if you put in the wrong dim array its gonna transpose...
        X = X'
    end
    XNew = zeros( Xsize[ 1 ] , Xsize[ 2 ] - 1)
    for c in 1 : ( Xsize[ 2 ] - 1 )
        XNew[ :, c ] = X[ :, c + 1 ] .- X[ :, c ]
    end
    return XNew
end

function SecondDerivative(X)
    Xsize = (length(size( X )) > 1) ? size( X ) : (1,length(X))
    if Xsize[1] == 1 #Sorry but if you put in the wrong dim array its gonna transpose...
        X = X'
    end
    #Xsize = (length(size( X )) > 1) ? size( X ) : (length(X),1)
    XNew = zeros( Xsize[ 1 ], Xsize[ 2 ] - 2 )
    for c in 1 : ( Xsize[ 2 ] - 2 )
        XNew[ :, c ] = (X[ :, c + 2 ] .- X[ :, c + 1 ]) - (X[ :, c + 1 ] .- X[ : , c ] )
    end
    return XNew
end

#Fractional Derivative via Grunwald Leitnikov algorithm,
#Useful for preprocessing data, and electrochemistry.
#Allows for nonuniform spacing in X...
function FractionalDerivative(Y, X = 1 : length(Y); Order = 0.5)
    (Obs,Vars) = size(Y)
    ddy = zeros(Obs,Vars-1)
    w = zeros(Vars)
    w[1] = 1.0
    for var in 2:Vars
        w[ var ] = w[ var - 1 ] * ( 1.0 - ( Order + 1.0 ) / ( var - 1.0 ) )
    end
    for var in 2:Vars
        h = (length(X) > 1) ? X[ var ] - X[ var - 1 ] : X
        for obs in 1 : Obs#This could definitely be broadcasted
            ddy[ obs, var-1 ] = w[ 1 : var ]' * Y[obs, var : -1: 1 ] ./ (h^Order)
        end
    end
    return ddy
end

#This is a little buggy, but its a start...
function SavitzkyGolay(X, Delta, PolyOrder, windowsize)
    @assert (windowsize % 2) == 1
    (Obs,Vars) = size(X)#length(Y
    windowspan = (windowsize - 1) / 2
    basis = ( ( -windowspan ) : windowspan ) .^ ( 0 : PolyOrder )'
    A = ( basis' * basis) \ basis'
    DeltaFac = factorial(Delta)
    output = DeltaFac * DSP.conv(X[1,:], A[Delta + 1,: ])'
    for r in 2:Obs
        output = vcat( output, DeltaFac * DSP.conv(X[r,:], A[Delta + 1,: ])' )
    end
    return output
end


#Direct Standardization Calibration Transfer Method
struct DirectStandardizationXform
    pca::PCA
    TransferMatrix::Array
end

function DirectStandardization(InstrumentX1, InstrumentX2; Factors = minimum(collect(size(InstrumentX1))) - 1)
    pcamodel = PCA(InstrumentX1; Factors = Factors)
    InstrumentX2_DS = pcamodel(InstrumentX2; Factors = Factors, inverse = false)
    TransferMatrix = LinearAlgebra.pinv(InstrumentX2_DS) * pcamodel.Scores[:, 1:Factors]
    return DirectStandardizationXform(pcamodel, TransferMatrix)
end

#Get the new samples in the Master instrument space...
function (DSX::DirectStandardizationXform)(X; Factors = length(DSX.pca.SingularValues))
    return X * DSX.pca.Loadings[1:Factors,:]' * DSX.TransferMatrix[1:Factors,1:Factors]  * DSX.pca.Loadings[1:Factors,:]
end

struct OrthogonalSignalCorrection
    Weights
    Loadings
    Residuals
end

function OrthogonalSignalCorrection(X, Y; Factors = 1)
    (Obs,Vars) = size(X)
    W = zeros(Vars, Factors)
    Loadings = zeros( Vars, Factors)
    Residuals = copy(X)
    M = LinearAlgebra.Diagonal(ones(Vars)) .- ( X' * Y * Base.inv(Y' * X * X' * Y) * Y' * X )
    P = svd( X * M )
    for factor in 1 : Factors
        Projection = M * Residuals' * ( P.U[ : , factor ] ./ P.S[ factor ] )
        W[ : , factor ] = Projection / sqrt(Projection' * Projection)
        t = Residuals * Projection
        p = Residuals' * (t ./ ( t' * t ))
        Loadings[ : , factor ] = p
        Residuals = Residuals .- (t * p')
    end
    return OrthogonalSignalCorrection(W, Loadings, Residuals)
end


function (OSC::OrthogonalSignalCorrection)(Z; Factors = 2)
    X = copy(Z)
    for factor in 1 : Factors
        X .-= (X * OSC.Weights[ : , factor ] * OSC.Loadings[:, factor ]')
    end
    return X
end

#Untested...
# struct TransferByOrthogonalProjection
#     pca::PCA
# end
#
# function TransferByOrthogonalProjection(X1, X2; Factors = 1)
#     (Obs,Vars) = size(X)
#     OneToTwo = PCA(X1 .- X2; Factors)
#     Instr1 = X1 * (LinearAlgebra.Diagonal( ones( Factors ) ) .- (OneToTwo.Loadings' * OneToTwo.Loadings))
#     return OrthogonalSignalCorrection(W, Loadings, Residuals)
# end
#
# function (TbOP::TransferByOrthogonalProjection)(X1; Factors = 1, inverse = false)
#     return X1 * (LinearAlgebra.Diagonal( ones( Factors ) ) .- (TbOP.pca.Loadings[:,1:Factors]' * TbOP.pca.Loadings[:,1:Factors]))
# end
