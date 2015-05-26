# The types defined below are used for specifying the options of the word embedding training
abstract Option

abstract InitializatioinMethod <: Option
type RandomInited <: InitializatioinMethod 
    # Initialize the embedding randomly
end
const random_inited = RandomInited()

abstract NetworkType <: Option
type NaiveSoftmax <: NetworkType
    # |V| outputs softmax
end
type HuffmanTree <: NetworkType
    # Predicate step by step on the huffman tree
end
const naive_softmax = NaiveSoftmax()
const huffman_tree = HuffmanTree()


type WordEmbedding
    vocabulary::Array{AbstractString}
    embedding::Dict{AbstractString, Array{Float64}}
    classification_tree::TreeNode
    distribution::Dict{AbstractString, Float64}
    codebook::Dict{AbstractString, Vector{Int64}}

    init_type::InitializatioinMethod
    network_type::NetworkType
    dimension::Int64
    lsize::Int64    # left window size in training
    rsize::Int64    # right window size
    trained_times::Dict{AbstractString,Int64}
    trained_count::Int64
    corpus_size::Int64
    subsampling::Float64
    init_learning_rate::Float64
    iter::Int64
    min_count::Int64
end

function WordEmbedding(dim::Int64, init_type::InitializatioinMethod, network_type::NetworkType; lsize=5, rsize=5, subsampling=1e-5, init_learning_rate=0.025, iter=5, min_count=5)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    WordEmbedding(AbstractString[], 
                    Dict{AbstractString,Array{Float64}}(),
                    nullnode,
                    Dict{AbstractString,Array{Float64}}(),
                    Dict{AbstractString,Vector{Int64}}(),
                    init_type, network_type,
                    dim,
                    lsize, rsize,
                    Dict{AbstractString,Int64}(),
                    0, 0,
                    subsampling, init_learning_rate, iter, min_count)
end

function Base.show(io::IO, x::WordEmbedding)
    println(io, "Word embedding(dimension = $(x.dimension)) of $(length(x.vocabulary)) words, trained on $(x.trained_count) words")
    nothing
end

# strip embedding and retain only word vectors
function _strip(embed::WordEmbedding)
    embed.vocabulary = AbstractString[]
    embed.classification_tree = nullnode
    embed.distribution = Dict{AbstractString,Array{Float64}}()
    embed.codebook = Dict{AbstractString,Vector{Int64}}()
    embed
end

# print the code book for debug
function _print_codebook(embed::WordEmbedding, N=10)
    for (word,code) in embed.codebook
        println("$word => $code")
        N -= 1
        (N > 0) || break
    end
    nothing
end

#===============================================================================
Step 1: Find Word Distribution
===============================================================================#

function merge_distributions(dst_ref::RemoteRef, src_ref::RemoteRef)
    word_count1,distribution1 = fetch(dst_ref)
    word_count2,distribution2 = fetch(src_ref)

    for (n,v) in distribution2
        if haskey(distribution1, n)
            distribution1[n] += v
        else
            distribution1[n] = v
        end
    end

    ((word_count1 + word_count2), distribution1)
end

merge_distributions(refs) = merge_distributions(convert(Array{RemoteRef,1}, refs))
function merge_distributions(refs::Array{RemoteRef,1})
    result = refs
    while length(result) > 1
        iter_result = Array(RemoteRef, 0)
        while length(result) > 1
            refpair = splice!(result, 1:2)
            push!(iter_result, remotecall(refpair[1].where, merge_distributions, refpair[1], refpair[2]))
        end
        isempty(result) || push!(iter_result, pop!(result))
        result = iter_result
    end
    fetch(result[1])
end

function get_distribution(corpus_fileio::IO)
    distribution = Dict{AbstractString,Float64}()
    word_count = 0

    for i in words_of(corpus_fileio)
        if !haskey(distribution, i)
            distribution[i] = 1
        else
            distribution[i] += 1
        end
        word_count += 1
    end

    (word_count, distribution)
end

function get_distribution(corpus_filename::AbstractString)
    open(corpus_filename, "r") do fs
        return get_distribution(fs)
    end
end

function get_distribution(b::Block)
    count_refs = pmap(get_distribution, b; fetch_results=false)
    merge_distributions(count_refs)
end

function strip_infrequent(distribution::Dict{AbstractString,Float64}, min_count::Int)
    stripped_distr = Dict{AbstractString,Float64}()
    word_count = 0

    for (k,v) in distribution
        if v >= min_count
            word_count += @compat(Int(v))
            stripped_distr[k] = v
        end
    end

    (word_count, stripped_distr)
end

function compute_frequency!(distribution::Dict{AbstractString,Float64}, word_count::Int)
    for (k, v) in distribution
        distribution[k] /= word_count
    end
    nothing
end

function word_distribution(source::Union(Block,AbstractString), min_count::Int=5)
    t1 = time()

    println("Finding word distribution...")
    word_count, distribution = get_distribution(source)
    println("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

    println("Stripping infrequent words...")
    word_count, distribution = strip_infrequent(distribution, min_count)
    println("Word Count: $word_count, Vocabulary Size: $(length(keys(distribution)))")

    compute_frequency!(distribution, word_count)
    println("Compute time: $(time()-t1)")

    distribution
end


#===============================================================================
Step 2: Calculate Word Embedding
===============================================================================#

function merge_embeds(ref1::RemoteRef, ref2::RemoteRef)
    e1 = fetch(ref1)
    e2 = fetch(ref2)
    for word in e1.vocabulary
        e1.embedding[word] .+= e2.embedding[word]
    end
    e1.trained_count += e2.trained_count
    println("merged $ref1 and $ref2")
    e1
end

merge_embeds(refs) = merge_embeds(convert(Array{RemoteRef,1}, refs))
function merge_embeds(refs::Array{RemoteRef,1})
    n = length(refs)
    result = refs
    while length(result) > 1
        iter_result = Array(RemoteRef, 0)
        while length(result) > 1
            refpair = splice!(result, 1:2)
            push!(iter_result, remotecall(refpair[1].where, merge_embeds, refpair[1], refpair[2]))
        end
        isempty(result) || push!(iter_result, pop!(result))
        result = iter_result
    end

    emb = fetch(result[1])
    for word in emb.vocabulary
        emb.embedding[word] /= n
    end

    emb
end

function work_process(ser_embed::AbstractString, words_stream::WordStream, strip::Bool=false)
    embed = restore(ser_embed)
    work_process(embed, words_stream, strip)
end

function work_process(embed::WordEmbedding, words_stream::WordStream, strip::Bool=false)
    t1 = time()
    middle = embed.lsize + 1
    input_gradient = zeros(Float64, embed.dimension)
    α = embed.init_learning_rate
    trained_count = 0
    trained_times = Dict{String, Int64}()

    for current_iter in 1:embed.iter
        for window in sliding_window(words_stream, lsize=embed.lsize, rsize=embed.rsize)
            trained_word = window[middle]
            trained_times[trained_word] = get(trained_times, trained_word, 0) + 1
            trained_count += 1

            if trained_count % 10000 == 0
                flen = words_stream.endpoint - words_stream.startpoint
                iter_progress = (position(words_stream.fp) - words_stream.startpoint) / flen
                progress = ((current_iter - 1) + iter_progress) / embed.iter
                println("trained on $trained_count words, progress $progress, α = $α")
                α = embed.init_learning_rate * (1 - progress)
                if α < embed.init_learning_rate * 0.0001
                    α = embed.init_learning_rate * 0.0001
                end
            end

            local_lsize = @compat(Int(rand(Uint64) % embed.lsize))
            local_rsize = @compat(Int(rand(Uint64) % embed.rsize))

            for ind in (middle - local_lsize) : (middle + local_rsize)
                (ind == middle) && continue

                target_word = window[ind]
                # discard words not presenting in the classification tree
                (haskey(embed.codebook, target_word) && haskey(embed.codebook, trained_word)) || continue

                node = embed.classification_tree::TreeNode

                fill!(input_gradient, 0.0)
                input = embed.embedding[trained_word]

                for code in embed.codebook[target_word]
                    train_one(node.data, input, code, input_gradient, α)
                    node = node.children[code]
                end

                for i in 1:embed.dimension
                    input[i] -= input_gradient[i]
                end
            end
        end
    end

    embed.trained_count = trained_count
    embed.trained_times = trained_times

    t2 = time()
    println("Finished training. Trained on $(embed.trained_count) words in $(t2-t1) seconds.")

    # strip to remove unnecessary members and make serialization faster
    strip && _strip(embed)

    embed
end


#===============================================================================
Train a corpus
Step 1: Find Word Distribution
Step 2: Calculate Word Embedding
===============================================================================#

function initialize_embedding(embed::WordEmbedding, randomly::RandomInited)
    for i in embed.vocabulary
        embed.embedding[i] = rand(1, embed.dimension) * 2 - 1
    end
    embed
end

function initialize_network(embed::WordEmbedding, huffman::HuffmanTree)
    heap = PriorityQueue()
    for (word, freq) in embed.distribution
        node = BranchNode([], word, nothing)    # the data field of leaf node is its corresponding word.
        enqueue!(heap, node, freq)
    end
    while length(heap) > 1
        (node1, freq1) = Base.Collections.peek(heap)
        dequeue!(heap)
        (node2, freq2) = Base.Collections.peek(heap)
        dequeue!(heap)
        newnode = BranchNode([node1, node2], LinearClassifier(2, embed.dimension), nothing) # the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    embed.classification_tree = dequeue!(heap)
    embed
end

function train(embed::WordEmbedding, corpus::Block, prepared_corpus_location::AbstractString)
    corpus = corpus |> as_io |> as_wordio
    embed.distribution = word_distribution(corpus)
    embed.vocabulary = collect(keys(embed.distribution))

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    # serialize the word distribution for workers to pick up
    save(embed, prepared_corpus_location)

    t1 = time()
    println("Starting parallel training...")

    # Note: subsampling is not honored here, probably not required also?
    corpus = corpus |> words_of

    embs = pmap((x)->work_process(prepared_corpus_location, x, true), corpus; fetch_results=false)
    t2 = time()
    println("Partial training done at $(t2-t1) time")

    println("Merging results...")
    training_result = merge_embeds(embs)
    embed.embedding = training_result.embedding
    embed.trained_count = training_result.trained_count
    t3 = time()

    println("Training complete at $(t3-t1) time")
    embed
end

function train(embed::WordEmbedding, corpus_filename::AbstractString)
    embed.distribution = word_distribution(corpus_filename)
    embed.vocabulary = collect(keys(embed.distribution))

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    t1 = time()
    println("Starting sequential training...")
    words_stream = words_of(corpus_filename, subsampling = (embed.subsampling, true, embed.distribution))
    work_process(embed, words_stream, false)

    t2 = time()
    println("Training complete at $(t2-t1) time")
    embed
end
