struct Bounds
    lower::Array{Float64, 1}
    upper::Array{Float64, 1}
end

"""
    Bounds(dims)

Default constructor for a Bounds object. Returns a bounds object with a lower bound of [0...] and upper bound[1...]
with length of `dims`.
"""
Bounds(dims) = Bounds( repeat( [ 0 ], dims ), repeat( [ 1 ], dims ) )

"""
    Bounds(dims)

Constructor for a Bounds object. Returns a bounds object with a lower bound of [lower...] and upper bound[upper...]
with length of `dims`.
"""
Bounds(lower, upper, dims) = Bounds( repeat( [ lower ], dims ), repeat( [ upper ], dims ) )

mutable struct Particle
    BestPos::Array{Float64, 1}
    BestScore::Float64
    Pos::Array{Float64, 1}
    Vel::Array{Float64, 1}
end

"""
    Particle(ProblemBounds, VelocityBounds)

Default constructor for a Particle object. It creates a random unformly distributed particle within the specified `ProblemBounds`,
and limits it's velocity to the specified `VelocityBounds`.
"""
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
"""
    PSO(fn, Bounds, VelRange, Particles; tolerance = 1e-6, maxiters = 1000, InertialDecay = 0.5, PersonalWeight = 0.5, GlobalWeight = 0.5, InternalParams = nothing)

Minimizes function `fn` with-in the user specified `Bounds` via a Particle Swarm Optimizer.
The particle velocities are limitted to the `VelRange`.
The number of particles are defined by the `Particles` parameter.

Returns a Tuple of the following form: ( GlobalBestPos, GlobalBestScore, P )
Where P is an array of the particles used in the optimization.

*Note: if the optimization function requires an additional constant parameter, please pass that parameter to InternalParams.
This will only work if the optimized parameter(o) and constant parameter(c) for the function of interest has the following format: F(o,c) *

Kennedy, J.; Eberhart, R. (1995). Particle Swarm Optimization. Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942–1948. doi:10.1109/ICNN.1995.488968
"""
function PSO(fn, Bounds::Bounds, VelRange::Bounds, Particles::Int;
                tolerance = 1e-6, maxiters = 1000,
                InertialDecay = 0.5, PersonalWeight = 0.5, GlobalWeight = 0.5, InternalParams = nothing)
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
            Score = isa(InternalParams, Nothing) ? fn(P[p].Pos) : fn(P[p].Pos, InternalParams)
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
