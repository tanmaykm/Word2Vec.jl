# linear softmax classifier (with stochastic gradient descent)
type LinearClassifier
    k::Int64 # number of outputs
    n::Int64 # number of inputs
    weights::Array{Float64, 2} # n * k weight matrix

    outputs :: Vector{Float64}
end

function LinearClassifier(k, n)
    weights = rand(n, k) * 2 - 1; # range is [-1, 1]
    LinearClassifier(k, n, weights, zeros(k))
end

function predict(c::LinearClassifier, x::Array{Float64})
    # the softmax() function from the NumericExtension package is more numeric stable
    return vec(softmax(x * c.weights))
end

function predict!(c::LinearClassifier, x::Array{Float64})
    # c.outputs = vec(softmax(x * c.weights))
    s = 0.0
    for i in 1:c.k
        s = 0.0
        for j in 1:c.n
            s += x[j] * c.weights[j, i]
        end
        c.outputs[i] = s
    end

    softmax!(c.outputs, c.outputs);
end

function train_one(c::LinearClassifier, x::Array{Float64}, y::Int64, α::Float64=0.025)
    # if !in(y, 1 : c.k)
    #     msg = @sprintf "A sample is discarded because the label y = %d is not in range of 1 to %d" y c.k
    #     warn(msg)
    #     return
    # end

    predict!(c, x)
    c.outputs[y] -= 1

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights)
    m = 0.0
    j = 0
    limit = c.n - 4
    for i in 1:c.k
        m = α * c.outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->c.weights[j + idx - 1, i] -= m * x[j + idx - 1])
            j+=4
        end
        while j <= c.n
            c.weights[j, i] -= m * x[j]
            j+=1
        end
    end
end

function train_one(c::LinearClassifier, x::Array{Float64}, y::Int64, input_gradient::Array{Float64}, α::Float64=0.025)
    predict!(c, x)
    c.outputs[y] -= 1

    # input_gradient = ( c.weights * outputs' )'
    # BLAS.gemv!('N', α, c.weights, c.outputs, 1.0, input_gradient)
    m = 0.0
    j = 0
    limit = c.n - 4
    for i in 1:c.k
        m = α * c.outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->input_gradient[j+idx-1] += m * c.weights[j+idx-1, i])
            j+=4
        end
        while j <= c.n
            input_gradient[j] += m * c.weights[j, i]
            j+=1
        end
    end

    # c.weights -= α * x' * outputs;
    # BLAS.ger!(-α, vec(x), c.outputs, c.weights)
    for i in 1:c.k
        m = α * c.outputs[i]
        j = 1
        while j <= limit
            @nexprs 4 (idx->c.weights[j + idx - 1, i] -= m * x[j + idx - 1])
            j+=4
        end
        while j <= c.n
            c.weights[j, i] -= m * x[j]
            j+=1
        end
    end
end

# calculate the overall log likelihood. Mainly used for debugging
function log_likelihood(c, X, y)
    n = size(X, 1)
    l = 0
    for i in 1:n
        l += log(predict(c, X[i, :])[y[i]])
    end
    return l
end

# calculate the accuracy on the testing dataset
function accuracy(c::LinearClassifier, X::Array{Float64}, y::Array{Int64})
    n = size(X, 1)
    succ = 0
    for i in 1 : n
        output = predict(c, X[i, :])
        if maximum(output) == output[y[i]]
            succ += 1
        end
    end
    return succ / n
end
