using Word2Vec
using Base.Test

data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
train_file = joinpath(data_dir, "mnist_train.csv")
test_file = joinpath(data_dir, "mnist_test.csv")

function test_softmax()
    println("Testing the softmax classifier on the MNIST dataset")

    println("Loading...")
    D = readcsv(train_file, header=true)[1]
    X_train = D[:, 2:end] / 255
    y_train = map(Int64, D[:,1] + 1)

    D = readcsv(test_file, header=true)[1]
    X_test = D[:, 2:end] / 255
    y_test= map(Int64, D[:, 1] + 1)

    println("Start training...")
    c = LinearClassifier(10, 784)
    niter = 20
    for j in 1:niter
        println("iteration $(j)/$(niter)")
        for i in 1:size(X_train,1)
            Word2Vec.train_one(c, X_train[i,:], y_train[i])
        end
        acc = accuracy(c, X_test, y_test)
        println("Accuracy on test set $acc (a value around 0.9 is expected)")
    end
end

test_softmax()
