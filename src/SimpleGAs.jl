mutable struct BinaryLifeform
    score
    genes
end

Lifeform(size, onlikelihood, initialscore) = BinaryLifeform(initialscore, rbinomial(onlikelihood, size))

function SinglePointCrossOver( L1::BinaryLifeform, L2::BinaryLifeform )
    len = length( L1.genes )
    @assert( len .== length( L2.genes ) )
    L = rand( 1 : ( len - 1 ) )
    return ( BinaryLifeform( -Inf, vcat(L1.genes[1:L], L2.genes[ ( L + 1 ) : end]) ),
             BinaryLifeform( -Inf, vcat(L2.genes[1:L], L1.genes[ ( L + 1 ) : end]) )  )
end

function Mutate( L::BinaryLifeform, amount = 0.05 )
    len = length(L.genes)
    mutater = rbinomial( amount, len ) .== 1
    L.genes[ mutater ] .= ( 1.0 .- L.genes[ mutater ] )
end
