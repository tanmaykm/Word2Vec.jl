const FILENAME = isempty(ARGS) ? "text8" : ARGS[1]
const MODELFILE = "$(FILENAME).model"
const NPROCS = (length(ARGS) > 1) ? parse(Int, ARGS[2]) : 4

addprocs(NPROCS)

using Word2Vec
using Base.Test
using Blocks
using Base.FS

function parallel_word_embedding(filename="text8", savemodel="")
    b = Block(File(filename), nworkers())
    embed = WordEmbedding(100, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0)
    @time train(embed, b)
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
        #inp = filter(x->!isempty(x), [strip(x) for x in split(l)])
        #nwords = find_nearest_words(embed, inp)
        #println("\t$(join(inp,',')) - $(join(nwords,','))")
    end
end

embed = isfile(MODELFILE) ? restore(MODELFILE) : parallel_word_embedding(FILENAME, MODELFILE)
show(embed)
test_nearest_words(embed)
interactive_nearest_words(embed)
