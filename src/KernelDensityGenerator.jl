mutable struct Universe
    min::Float64
    max::Float64
    width::Float64
    halfwidth::Float64
    bins::Int
    spectra::Array{Float64,1}
end

"""
    Universe(mini, maxi; width = nothing, bins = nothing)

Creates a 1-D discretized segment that starts at mini and ends at maxi. The width of the bins for the discretization
can either be provided or inferred from the number of bins. Returns a Universe object.
"""
function Universe(mini, maxi; width = nothing, bins = nothing)
    nowidth = isa( width, Nothing )
    nobins =  isa( bins, Nothing )
    @assert ( nowidth + nobins ) < 2
    if nowidth
        width = (maxi - mini ) / bins
    end
    if nobins
        bins = round((maxi - mini ) / width) |> Int
    end
    return Universe( mini, maxi, width, width / 2.0, bins, zeros( bins ) )
end

"""
    SpectralArray(Universes::Array{Universe,1})

Takes an array of Universe types and returns a 2-Array of the spectra.
"""
function SpectralArray(Universes::Array{Universe,1})
    RetArray = zeros( length(Universes), length(Universes[1].spectra) )
    for (spectra, uni) in enumerate(Universes)
        RetArray[ spectra, : ] = uni.spectra
    end
    return RetArray
end

"""
    GaussianBand(sigma,amplitude,center)

Constructs a Gaussian kernel generator.
"""
struct GaussianBand
    sigma::Float64
    amplitude::Float64
    center::Float64
end

"""
    (B::GaussianBand)(X::Float64)

Returns the scalar probability associated with a GaussianBand object (kernel) at a location in space(`X`).
"""
(B::GaussianBand)(X::Float64) = B.amplitude * exp( -0.5 * ( ( X - B.center ) / B.sigma) ^ 2 ) / sqrt( 2.0 * pi * B.sigma )

"""
    LorentzianBand(gamma,amplitude,center)

Constructs a Lorentzian kernel generator.
"""
struct LorentzianBand
    gamma::Float64
    amplitude::Float64
    center::Float64
end

"""
    (B::LorentzianBand)(X::Float64)

Returns the probability associated with a LorentzianBand object (kernel) at a location in space(`X`).
"""
(B::LorentzianBand)(X::Float64) = B.amplitude / ( pi * B.gamma * (1 + ( ( X - B.center ) / B.gamma) ^ 2 ) )

"""
    (U::Universe)(Band::Union{ GaussianBand, LorentzianBand})

A Universe objects internal "spectra" can be updated to include the additive contribution of any Band-like object.
"""
function (U::Universe)(Band::Union{ GaussianBand, LorentzianBand})
    for b in 1 : U.bins
        U.spectra[b] += Band( U.min + ( b * U.width ) + U.halfwidth )
    end
end

"""
    (U::Universe)(Band...)

A Universe objects internal "spectra" can be updated to include the additive contribution of many Band-like objects.
"""
function (U::Universe)(Band...)
    for bin in 1 : U.bins, band in Band
        U.spectra[bin] += band( U.min + ( bin * U.width ) + U.halfwidth )
    end
end
