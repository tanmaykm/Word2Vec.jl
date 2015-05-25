module Word2Vec

using Base.Collections      # for priority queue
using Base.Cartesian        # for @nexprs
using Distances
using NumericExtensions
using Blocks

export LinearClassifier, train_one, WordEmbedding, train, accuracy
#export train_parallel
export save, restore
export find_nearest_words

include("utils.jl")
include("tree.jl")
include("word_stream.jl")
include("softmax_classifier.jl")
include("train.jl")
include("query.jl")

end # module
