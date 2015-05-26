module Word2Vec

using Base.Collections      # for priority queue
using Base.Cartesian        # for @nexprs
using Distances
using NumericExtensions
using Blocks
using Compat

if isless(Base.VERSION, v"0.4.0-")
using Iterators
end

export LinearClassifier, train_one, WordEmbedding, train, accuracy
export save, restore
export find_nearest_words

include("utils.jl")
include("tree.jl")
include("word_stream.jl")
include("softmax_classifier.jl")
include("train.jl")
include("query.jl")

end # module
