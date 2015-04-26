using Word2Vec

function test_word_window()
    fp = open("test_text")
    for i in words_of(fp)
        println(i)
        println("--------")
    end

    seekstart(fp)
    for i in imap(uppercase, words_of(fp))
        println(i)
    end

    seekstart(fp)
    for i in sliding_window(words_of(fp))
        println(i)
    end
end

function test_softmax()
    @printf "Testing the softmax classifier on the MNIST dataset\n"
    @printf "Loading...\n"
    D = readcsv("mnist_train.csv")
    X_train = D[:, 2:end] / 255
    y_train = Int64(D[:, 1] + 1)

    D = readcsv("mnist_test.csv")
    X_test = D[:, 2:end] / 255
    y_test= Int64(D[:, 1] + 1)

    @printf "Start training...\n"
    c = LinearClassifier(10, 784)
    @time train_parallel(c, X_train, y_train, max_iter = 10)
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
