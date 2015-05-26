using Compat

data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
model_dir = joinpath(Pkg.dir("Word2Vec"), "test", "models")

test_filename = isempty(ARGS) ? "text8" : ARGS[1]
test_file = joinpath(data_dir, test_filename)
model_file = joinpath(model_dir, test_filename * ".model")

np = (length(ARGS) > 1) ? @compat(parse(Int, ARGS[2])) : 4
addprocs(np)

@everywhere using Word2Vec
@everywhere using Base.Test
@everywhere using Blocks
@everywhere using Base.FS
@everywhere using Compat

function parallel_word_embedding(filename, savemodel="")
    b = Block(File(filename), nworkers())
    embed = WordEmbedding(100, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 1e-4)
    @time train(embed, b, "/tmp/emb")
    isempty(savemodel) || save(embed, savemodel)
    embed
end

function test_nearest_words(embed)
    println("nearest words:")
    inp = ["king", "queen", "prince"]
    for w in inp
        nwords = find_nearest_words(embed, w)
        println("\t$w - $(join(nwords,','))")
    end
    nwords = find_nearest_words(embed, join(inp, ' '))
    println("\t$(join(inp,',')) - $(join(nwords,','))")
end

function interactive_nearest_words(embed)
    cont = true
    while cont
        println("\nEnter words to find nearest word:")
        l = strip(readline(STDIN))
        nwords = find_nearest_words(embed, strip(l))
        println("\t$(l) - $(join(nwords,','))")
    end
end

function restore_or_train()
    if isfile(model_file)
        restore(model_file)
    else
        parallel_word_embedding(test_file, model_file)
    end
end

embed = restore_or_train()
show(embed)
test_nearest_words(embed)
interactive_nearest_words(embed)
