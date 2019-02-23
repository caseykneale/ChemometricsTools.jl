struct Bounds
    lower::Array{Float64, 1}
    upper::Array{Float64, 1}
end
Bounds(dims) = Bounds( repeat( [ 0 ], dims ), repeat( [ 1 ], dims ) )
Bounds(lower, upper, dims) = Bounds( repeat( [ lower ], dims ), repeat( [ upper ], dims ) )

mutable struct Particle
    BestPos::Array{Float64, 1}
    BestScore::Float64
    Pos::Array{Float64, 1}
    Vel::Array{Float64, 1}
end

function Particle(ProblemBounds, VelocityBounds)
    Dimen = length(ProblemBounds.upper)
    @assert Dimen == length(VelocityBounds.upper)
    BndDists = ProblemBounds.upper .- ProblemBounds.lower
    VelDists = VelocityBounds.upper .- VelocityBounds.lower
    PosI = (BndDists .* rand(Dimen)) .+ ProblemBounds.lower
    VelI = (VelDists .* rand(Dimen)) .+ VelocityBounds.lower
    return Particle( PosI, Inf, PosI, VelI )
end

#Particle( bounds(2,-1,3), bounds(0,1,3) )

#This is a vanilla PSO minimizer... make your fn negative to maximize...
function PSO(fn, Bounds, VelRange, Particles;
                tolerance = 1e-6, maxiters = 1000,
                InertialDecay = 0.5, PersonalWeight = 0.5, GlobalWeight = 0.5)
    P = [ Particle(Bounds, VelRange) for x in 1 : Particles ]
    Dimen = length(Bounds.upper)
    GlobalBestPos = P[1].Pos
    GlobalBestScore = Inf
    iter = 0
    while (GlobalBestScore >= tolerance) && (iter < maxiters)
        for p in 1 : Particles
            if iter > 1
                #Update Positions
                (Wpersonal, Wglobal) = ( rand( Dimen ), rand( Dimen ) )
                P[p].Vel .=  ( InertialDecay .* P[p].Vel ) .+
                          ( Wpersonal .* PersonalWeight .* ( P[p].BestPos .-  P[p].Pos) ) .+
                          ( Wglobal   .* GlobalWeight   .* ( GlobalBestPos .- P[p].Pos ) )
                P[p].Pos .+= P[p].Vel
            end
            outbndslow = Bounds.lower .>= P[p].Pos
            if any(outbndslow)
                P[ p ].Pos[ outbndslow ] .= Bounds.lower[ outbndslow ]
            end
            outbndshigh = P[p].Pos .>= Bounds.upper
            if any(outbndshigh)
                P[ p ].Pos[ outbndshigh ] .= Bounds.upper[ outbndshigh ]
            end
            #Evaluate Fn to Obtain Score
            Score = fn(P[p].Pos)
            #Evaluate Scores
            if Score < P[p].BestScore
                P[p].BestPos .= P[p].Pos; P[p].BestScore = Score
                if Score < GlobalBestScore
                    GlobalBestScore = Score; GlobalBestPos = P[p].Pos
                end
            end
        end
        iter += 1
    end
    return ( GlobalBestPos, GlobalBestScore, P )
end

#parab(x) = sum(x.^2)

#PSO( parab, bounds(-3, 3, 1), bounds(0, 1, 1), 10;
#                tolerance = 1e-9, maxiters = 1000,
#                InertialDecay = 0.5, PersonalWeight = 0.5, GlobalWeight = 0.5)
