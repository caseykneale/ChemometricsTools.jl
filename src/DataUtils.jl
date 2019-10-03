#This will contain convenience functions for manipulating data.
#Primary focus will be on scrubbing, splitting and preparing DataFrames
#for modelling.
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
