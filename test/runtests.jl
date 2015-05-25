include("test_utils.jl")
include("test_softmax.jl")

addprocs(4)
include("test_embedding.jl")
