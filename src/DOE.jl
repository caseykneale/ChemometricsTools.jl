"""
    SimplexLatticeDesign( Components::Int, Spaces::Int )

Returns an array of tuples (`Components` in length) which represent design points.
"""
function SimplexLatticeDesign( Components::Int, Spaces::Int )
    @assert( ( Components > 0 ) && ( Spaces > 0 ), "Components and Spaces must be positive integers!" )
    DesignPoints = Inf
    try
        DesignPoints = Int( factorial( Components + Spaces - 1) / ( factorial(Spaces) * factorial(Components - 1) ))
    catch
        @warn("Cannot estimate number of design points. May execute for the lifetime of the universe.")
    end
    PossiblePoints = reverse(0:Spaces) ./ Spaces
    DesignSpace = repeat([PossiblePoints], Components)
    Possible = [ prod for prod in Base.product(DesignSpace...) if (sum(prod) == 1) ]

    if !isinf(DesignPoints)
        @assert( DesignPoints == length(Possible) )
    end
    return Possible
end

"""
    SimplexCentroidDesign( Components::Int, Order::Union{ UnitRange{ Int }, Int} )

Returns an array of tuples (`Components` in length) which represent design points.

Please note: the Order can be an Int or a UnitRange.
"""
function SimplexCentroidDesign( Components::Int, Order::Union{ UnitRange{ Int }, Int} )
    HighestOrder = Order[end]
    @assert( ( Components > 0 ) && ( HighestOrder > 0 ), "Components and the highest orders must be positive integers!" )
    @assert( HighestOrder <= Components, "The highest order of the design must be less than the number of components." )
    DesignPoints = Inf
    try
        DesignPoints = 2 ^ ( Components ) - 2 ^ ( Components - HighestOrder )
        if isa(Order, Int)
            DesignPoints = 2 ^ ( HighestOrder ) - 1
        end
    catch
        @warn("Cannot estimate number of design points. May execute for the lifetime of the universe.")
    end

    PossiblePoints = zeros( Components )
    DesignSpace = []
    for order in Order
        PossiblePoints[ 1:order ] .= 1.0 / order
        push!( DesignSpace, unique( collect( Combinatorics.permutations(PossiblePoints, Components) ) ) )
    end
    DesignSpace = ( isa(Order, Int) ) ? DesignSpace[1] : reduce(vcat, DesignSpace)
    if !isinf(DesignPoints)
        println( DesignPoints )
        println(length(DesignSpace))
        @assert( DesignPoints == length(DesignSpace) )
    end
    return DesignSpace
end


# """
# The Sequential Generation of D-Optimum Experimental Designs. Wynn. The Annals of Mathematical Statistics. 41(5). 1970
# """
# function FedorovWynn(Components::Int, Points::Int)
#     X = rand( Points, Components )
#     while ( 1 / LinearAlgebra.cond(X' * X) ) < 1e-6
#         X = rand( Points, Components )
#     end
#     M = X' * X
#     Minv = Base.inv( M )
#     f(x) = sum(x)
# end
