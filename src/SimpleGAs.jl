mutable struct BinaryLifeform
    score
    genes
end

"""
    Lifeform(size, onlikelihood, initialscore)

Constructor for a BinaryLifeForm struct. Binary life forms are basically wrappers for a binary vector, which
has a likelihood for being 1(`onlikelihood`). Each life form also has a `score` based on it's "fitness". So
the GA's in this package can be used to minimize or maximize this is an open parameter, but Inf/-Inf is a good
`initialscore`.
"""
Lifeform(size, onlikelihood, initialscore) = BinaryLifeform(initialscore, rbinomial(onlikelihood, size))

"""
    SinglePointCrossOver( L1::BinaryLifeform, L2::BinaryLifeform )

Creates two offspring (new BinaryLifeForms) by mixing the genes from `L1` and `L2` after a random position in the vector.
"""
function SinglePointCrossOver( L1::BinaryLifeform, L2::BinaryLifeform )
    len = length( L1.genes )
    @assert( len .== length( L2.genes ) )
    L = rand( 1 : ( len - 1 ) )
    return ( BinaryLifeform( -Inf, vcat(L1.genes[1:L], L2.genes[ ( L + 1 ) : end]) ),
             BinaryLifeform( -Inf, vcat(L2.genes[1:L], L1.genes[ ( L + 1 ) : end]) )  )
end


"""
    Mutate( L::BinaryLifeform, amount = 0.05 )

Assesses each element in the gene vector (inside of `L`). If a randomly drawn value has a binomial probability
of `amount` the element is mutated.
"""
function Mutate( L::BinaryLifeform, amount = 0.05 )
    len = length(L.genes)
    mutater = rbinomial( amount, len ) .== 1
    L.genes[ mutater ] .= ( 1.0 .- L.genes[ mutater ] )
end
