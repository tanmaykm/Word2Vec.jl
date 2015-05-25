
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

function work_process(ser_embed::AbstractString, words_stream::WordStream)
    embed = restore(ser_embed)
    work_process(embed, words_stream)
end

# now we can start the training
function work_process(embed::WordEmbedding, words_stream::WordStream)
    t1 = time()
    if isfile("/tmp/emb-$(myid())")
        embed = restore("/tmp/emb-$(myid())")
        t2 = time()
        println("Restored training. Restored training data on $(embed.trained_count) words in $(t2-t1) seconds. Sending result to the main process")
        return embed
    end
    middle = embed.lsize + 1
    input_gradient = zeros(Float64, embed.dimension)
    α = embed.init_learning_rate
    trained_count = 0
    trained_times = Dict{String, Int64}()

    for current_iter in 1:embed.iter
        for window in sliding_window(words_stream, lsize = embed.lsize, rsize = embed.rsize)
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

            local_lsize = Int(rand(Uint64) % embed.lsize)
            local_rsize = Int(rand(Uint64) % embed.rsize)

            for ind in (middle - local_lsize) : (middle + local_rsize)
                if ind == middle
                    continue
                end
                target_word = window[ind]
                if !haskey(embed.codebook, target_word) || !haskey(embed.codebook, trained_word)
                    # discard words not presenting in the classification tree
                    continue
                end
                # @printf "%s -> %s\n" trained_word target_word
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
    println("Finished training. Trained on $(embed.trained_count) words in $(t2-t1) seconds. Sending result to the main process")
    _strip(embed)
    save(embed, "/tmp/emb-$(myid())")
    t3 = time()
    println("Saved result at /tmp/emb/$(myid()) in $(t3-t2) seconds")
    embed
end

function _word_distribution(corpus_fileio::IO)
    distribution = Dict{AbstractString, Float64}()
    word_count = 0

    println("finding word distribution...")

    for i in words_of(corpus_fileio)
        if !haskey(distribution, i)
            distribution[i] = 1
        else
            distribution[i] += 1
        end
        word_count += 1
    end

    println("Word count: $word_count words")
    println("Vocabulary size: $(length(keys(distribution)))")

    word_count, distribution
end

function _strip_infrequent(distribution::Dict{AbstractString, Float64}, min_count::Int=5)
    dist2 = Dict{AbstractString, Float64}()
    word_count = 0
    for (k,v) in distribution
        if v >= min_count
            word_count += v
            dist2[k] = v
        end
    end
    (word_count, dist2)
end

function word_distribution(corpus_filename::AbstractString, min_count::Int=5)
    open(corpus_filename, "r") do fs
        word_count, distribution = _word_distribution(fs)
        word_count, distribution = _strip_infrequent(distribution, min_count)

        for (k, v) in distribution
            distribution[k] /= word_count
        end

        return distribution
    end
end

function _merge_distributions(ref1::RemoteRef, ref2::RemoteRef)
    wc1,d1 = fetch(ref1)
    wc2,d2 = fetch(ref2)
    for (n,v) in d2
        if haskey(d1, n)
            d1[n] += v
        else
            d1[n] = v
        end
    end
    (wc1+wc2), d1
end

function _merge_distributions(refs::Array{RemoteRef,1})
    result = Array(RemoteRef, 0)
    while length(refs) > 1
        refpair = splice!(refs, 1:2)
        push!(result, remotecall(refpair[1].where, _merge_distributions, refpair[1], refpair[2]))
    end
    isempty(refs) || push!(result, pop!(refs))
    result
end

merge_distributions(refs) = merge_distributions(convert(Array{RemoteRef,1}, refs))
function merge_distributions(refs::Array{RemoteRef,1})
    result = refs
    while length(result) > 1
        result = _merge_distributions(result)
    end
    fetch(result[1])
end

function word_distribution(b::Block, min_count::Int=5)
    t1 = time()
    count_refs = pmap(_word_distribution, b; fetch_results=false)
    word_count, distribution = merge_distributions(count_refs)
    println("Total Word Count: $word_count words")
    println("Total Vocabulary Size: $(length(keys(distribution)))")

    println("Stripping infrequent words...")
    word_count, distribution = _strip_infrequent(distribution, min_count)

    for (k, v) in distribution
        distribution[k] /= word_count
    end

    println("Total Word Count: $word_count words")
    println("Total Vocabulary Size: $(length(keys(distribution)))")
    println("Compute time: $(time()-t1)")

    distribution
end

function reduce_embeds!(embed, embs)
    n = length(embs)
    for word in embed.vocabulary
        wordembs = [emb.embedding[word] for emb in embs]
        embed.embedding[word] = .+(wordembs...) / n
    end
    embed.trained_count = sum(map(emb->emb.trained_count, embs))
    embed
end

function _merge_embeds(ref1::RemoteRef, ref2::RemoteRef)
    e1 = fetch(ref1)
    e2 = fetch(ref2)
    for word in e1.vocabulary
        e1.embedding[word] .+= e2.embedding[word]
    end
    e1.trained_count += e2.trained_count
    println("merged $ref1 and $ref2")
    e1
end

function _merge_embeds(refs::Array{RemoteRef,1})
    result = Array(RemoteRef, 0)
    while length(refs) > 1
        refpair = splice!(refs, 1:2)
        push!(result, remotecall(refpair[1].where, _merge_embeds, refpair[1], refpair[2]))
    end
    isempty(refs) || push!(result, pop!(refs))
    result
end

merge_embeds(refs) = merge_embeds(convert(Array{RemoteRef,1}, refs))
function merge_embeds(refs::Array{RemoteRef,1})
    n = length(refs)
    result = refs
    while length(result) > 1
        result = _merge_embeds(result)
    end
    emb = fetch(result[1])
    for word in emb.vocabulary
        emb.embedding[word] /= n
    end
    emb
end

function train(embed::WordEmbedding, corpus::Block)
    if isfile("/tmp/emb")
        corpus = corpus |> as_io |> as_wordio |> words_of
    else
        corpus = corpus |> as_io |> as_wordio
        embed.distribution = word_distribution(corpus)
        embed.vocabulary = collect(keys(embed.distribution))

        initialize_embedding(embed, embed.init_type)        # initialize by the specified method
        initialize_network(embed, embed.network_type)

        # determine the position in the tree for every word
        for (w, code) in leaves_of(embed.classification_tree)
            embed.codebook[w] = code
        end

        # Note: subsampling is not honored here, probably not required also?
        corpus = corpus |> words_of
        save(embed, "/tmp/emb")
    end
    t1 = time()
    println("Starting parallel training...")
    embs = pmap((x)->work_process("/tmp/emb", x), corpus; fetch_results=false)
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

    number_workers = nworkers()
    println("Starting parallel training...")
    words_streams = parallel_words_of(corpus_filename, number_workers, subsampling = (embed.subsampling, true, embed.distribution))
    embs = pmap(work_process, collect(repeated(embed, number_workers)), words_streams)
    println("Merging results...")
    reduce_embeds!(embed, embs)
    embed
end

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

function Base.show(io::IO, x::WordEmbedding)
    println(io, "Word embedding(dimension = $(x.dimension)) of $(length(x.vocabulary)) words, trained on $(x.trained_count) words")
    #for (word, embed) in take(x.embedding, 5)
    #    println(io, "$word => $embed")
    #end
    #if length(x.embedding) > 5
    #    println(io, "......")
    #end
end

function joinvec(embed::WordEmbedding, words::Vector, fn::Function, wv::Array=[])
    for word in words
        if !haskey(embed.embedding, word)
            warn("'$word' not present in the vocabulary")
            return wv
        end
        if isempty(wv)
            wv = vec(embed.embedding[word])
        else
            wv = fn(wv, vec(embed.embedding[word]))
        end
    end
    wv
end

function find_nearest_words(embed::WordEmbedding, words::AbstractString; k=5)
    positive_words = []
    negative_words = []
    wordlist = positive_words
    for tok in split(words)
        tok = strip(tok)
        isempty(tok) && continue
        if tok == "+"
            wordlist = positive_words
        elseif tok == "-"
            wordlist = negative_words
        else
            push!(wordlist, tok)
        end
    end
    find_nearest_words(embed, positive_words, negative_words; k=k)
end

#find_nearest_words(embed::WordEmbedding, word::AbstractString; k=5) = find_nearest_words(embed, [word], []; k=k)
function find_nearest_words(embed::WordEmbedding, positive_words::Vector, negative_words::Vector; k=5)
    pq = PriorityQueue(Base.Order.Reverse)

    wv = joinvec(embed, positive_words, .+, Array[])
    wv = joinvec(embed, negative_words, .-, wv)

    if !isempty(wv)
        for (w, embed_w) in embed.embedding
            ((w in positive_words) || (w in negative_words)) && continue
            dist = cosine_dist(wv, vec(embed_w))
            enqueue!(pq, w, dist)
            (length(pq) > k) && dequeue!(pq)
        end
    end
    sort(collect(pq), by = t -> t[2])
end
