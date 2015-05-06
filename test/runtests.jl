using Word2Vec
using Base.Test
using Blocks
using Base.FS


printover(i) = print("\r$i\e[K")
function test_word_window()
    fp = open("text8_tiny")
    println("Testing words_of...")
    for i in Word2Vec.words_of(fp)
        printover(i)
    end

    seekstart(fp)
    for i in map(uppercase, Word2Vec.words_of(fp))
        printover(i)
    end

    println("\rDone.\e[K")
    println("Testing sliding_window...")
    seekstart(fp)
    for i in Word2Vec.sliding_window(Word2Vec.words_of(fp))
        printover(join(i, ','))
    end
    println("\nDone.")
end

function test_softmax()
    @printf "Testing the softmax classifier on the MNIST dataset\n"
    @printf "Loading...\n"
    #D = readcsv("mnist_train.csv")
    D = readcsv("mnist_train.csv", header=true)[1]
    X_train = D[:, 2:end] / 255
    y_train = map(Int64, D[:,1] + 1)

    D = readcsv("mnist_test.csv", header=true)[1]
    X_test = D[:, 2:end] / 255
    y_test= map(Int64, D[:, 1] + 1)

    @printf "Start training...\n"
    c = LinearClassifier(10, 784)
    @time train_parallel(c, X_train, y_train, max_iter = 20)
    @printf "Accuracy on test set %f (a value around 0.9 is expected)\n" accuracy(c, X_test, y_test)
end

function test_word_embedding()
    embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0)
    @time train(embed, "text8_small")
    embed
end

function test_word_embedding_tiny()
    embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 0)
    @time train(embed, "text8_tiny")
    embed
end

function test_word_embedding_large()
    embed = WordEmbedding(30, Word2Vec.random_inited, Word2Vec.huffman_tree, subsampling = 1e-4)
    @time train(embed, "text8")
    embed
end

test_word_window()
test_softmax()

embed = test_word_embedding_tiny()
save(embed, "model")
embed = restore("model")

inp = ["king", "queen", "prince"]
for w in inp
    println("nearest words to $w")
    println(find_nearest_words(embed, w))
end
println("nearest words to $inp")
println(find_nearest_words(embed, ["king", "queen", "prince"]))
