module Word2Vec

using Base.Collections      # for priority queue
using Distances
using NumericExtensions

export LinearClassifier, train_one, train_parallel, WordEmbedding, train

include("utils.jl")
include("softmax_classifier.jl")
include("word_embedding.jl")

end # module
