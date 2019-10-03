TecatorStatement = "Statement of permission from Tecator (the original data source).These data are recorded on a Tecator" *
"\n Infratec Food and Feed Analyzer working in the wavelength range 850 - 1050 nm by the Near Infrared" *
"\n Transmission (NIT) principle. Each sample contains finely chopped pure meat with different moisture, fat" *
"\n and protein contents.If results from these data are used in a publication we want you to mention the" *
"\n instrument and company name (Tecator) in the publication. In addition, please send a preprint of your " *
"\n article to Karin Thente, Tecator AB, Box 70, S-263 21 Hoganas, Sweden. The data are available in the " *
"\n public domain with no responsability from the original data source. The data can be redistributed as long " *
"\n as this permission note is attached. For more information about the instrument - call Perstorp Analytical's" *
"\n representative in your area."

datapath = joinpath(@__DIR__, "..", "data")

"""
    ChemometricsToolsDatasets()

Displays a list of the available datasets in ChemometricsTools.jl .

"""
function ChemometricsToolsDatasets()
    dircontents = readdir(datapath)
    dircontents = [ f for f in dircontents if f != "Readme.md" ]
    return Dict( (1:length(dircontents)) .=> dircontents )
end

"""
    ChemometricsToolsDataset(file::Int)

Loads in a dataset included in the ChemometricsTools.jl package.

"""
function ChemometricsToolsDataset(filename::String)
    if filename == "tecator.csv"
        println(TecatorStatement)
    end
    if filename != "Readme.md"
        read( Base.joinpath( datapath, filename ) )
    else
        println("Don't load the markdown Readme as a csv... You're better than this.")
    end
end

"""
    ChemometricsToolsDataset(file::Int)

Loads in a dataset included in the ChemometricsTools.jl package.

"""
function ChemometricsToolsDataset(file::Int)
    if readdir(datapath)[file] == "tecator.csv"
        println(TecatorStatement)
    end
    read( Base.joinpath( datapath, readdir(datapath)[file] ) )
end

"""
    FindCommonVariables( DFArray::Array{DataFrame,1} )

Finds exact matches across variable names over an array of DataFrames.
Variables that are not mutual to each dataframe are discarded.

For intelligent/soft matching please see ExploreCommonVariables().

"""
function FindCommonVariables( DFArray::Array{DataFrame,1} )
    common = []
    for df in DFArray
        curvars = Set( names( df ) )
        common = ( length( common ) > 0) ? curvars : intersect(common, curvars)
    end
    return common
end

# function ExploreCommonVariables( DFArray::Array{DataFrame,1}, allowed_distance = 1 )
#     common_names = []
#     common_types = []
#     for df in DFArray
#         cur_names = names( df )
#         cur_types = types( df )
#         if length( common ) > 0
#             common_names = cur_names
#         else
#             common_names = intersect(common, cur_names)
#         end
#     end
#     return common_names, common_types
# end
