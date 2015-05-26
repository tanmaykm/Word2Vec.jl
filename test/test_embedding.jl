using Word2Vec
using Base.Test
using Blocks
using Base.FS

data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
model_dir = joinpath(Pkg.dir("Word2Vec"), "test", "models")

test_filename = isempty(ARGS) ? "text8_tiny" : ARGS[1]
# test_filename = "text8"
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")

subsampling = 0
#subsampling = 1e-4

function test_word_embedding(inputfile, subsampling)
    println("=======================================")
    println("Testing word embedding with $inputfile")
    println("=======================================")

    embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = subsampling)
    @time train(embed, inputfile)

    save(embed, model_file)
    embed = restore(model_file)

    inp = ["king", "queen", "prince"]
    for w in inp
        println("nearest words to $w")
        println(find_nearest_words(embed, w))
    end
    println("nearest words to $inp")
    println(find_nearest_words(embed, ["king", "queen", "prince"], []))
end

test_word_embedding(test_file, subsampling)
