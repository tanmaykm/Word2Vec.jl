
# The types defined below are used for specifying the options of the word embedding training
abstract Option

abstract InitializatioinMethod <: Option
type RandomInited <: InitializatioinMethod 
    # Initialize the embedding randomly
end
type OntologyInited <: InitializatioinMethod
    # Initialize the embedding by the structure of ontology tree
    ontology :: TreeNode
end
const random_inited = RandomInited()

abstract NetworkType <: Option
type NaiveSoftmax <: NetworkType
    # |V| outputs softmax
end
type HuffmanTree <: NetworkType
    # Predicate step by step on the huffman tree
end
type OntologyTree <: NetworkType
    # Predicate step by step on the ontology tree
    ontology :: TreeNode
end
const naive_softmax = NaiveSoftmax()
const huffman_tree = HuffmanTree()


type WordEmbedding
    vocabulary :: Array{AbstractString}
    embedding :: Dict{AbstractString, Array{Float64}}
    classification_tree :: TreeNode
    distribution :: Dict{AbstractString, Float64}
    codebook :: Dict{AbstractString, Vector{Int64}}

    init_type :: InitializatioinMethod
    network_type :: NetworkType
    dimension :: Int64
    lsize :: Int64    # left window size in training
    rsize :: Int64    # right window size
    trained_count :: Int64
    subsampling :: Float64
end

function WordEmbedding(dim :: Int64, init_type :: InitializatioinMethod, network_type :: NetworkType; lsize = 5, rsize = 5, subsampling = -1)
    if dim <= 0 || lsize <= 0 || rsize <= 0
        throw(ArgumentError("dimension should be a positive integer"))
    end
    return WordEmbedding(AbstractString[], Dict{AbstractString,Array{Float64}}(), nullnode, Dict{AbstractString,Array{Float64}}(), Dict{AbstractString,Vector{Int64}}(), init_type, network_type, dim, lsize, rsize, 0, subsampling);
end

# now we can start the training
function work_process(embed :: WordEmbedding, words_stream :: Task)
    middle = embed.lsize + 1
    input_gradient = zeros(Float64, embed.dimension)
    for window in sliding_window(words_stream, lsize = embed.lsize, rsize = embed.rsize)
        trained_word = window[middle]
        embed.trained_count += 1
        if embed.trained_count % 1000 == 0
            @printf "trained on %d words\n" embed.trained_count
        end
        local_lsize = Int(rand(Uint64) % embed.lsize)
        local_rsize = Int(rand(Uint64) % embed.rsize)
        # @printf "lsize: %d, rsize %d\n" local_lsize local_rsize
        for ind in (middle - local_lsize) : (middle + local_rsize)
            if ind == middle
                continue
            end
            target_word = window[ind]
            if !haskey(embed.codebook, target_word)
                # discard words not presenting in the classification tree
                continue;
            end
            # @printf "%s -> %s\n" trained_word target_word
            node = embed.classification_tree :: TreeNode
            fill!(input_gradient, 0.0)
            for code in embed.codebook[target_word]
                train_one(node.data, embed.embedding[trained_word], code, input_gradient)
                node = node.children[code]
            end
            BLAS.axpy!(embed.dimension, -1.0, vec(input_gradient), 1, vec(embed.embedding[trained_word]), 1)
        end
    end
    @printf "finished training, sending result to the main process\n"
    embed
end

function train(embed :: WordEmbedding, corpus_filename :: String)
    fs = open(corpus_filename, "r")
    distribution = Dict{AbstractString, Float64}()
    word_count = 0
    @printf "reading the file...\n"
    for i in words_of(fs)
        if !haskey(distribution, i)
            distribution[i] = 0
        end
        distribution[i] += 1
        word_count += 1
    end
    for (k, v) in distribution
        distribution[k] /= word_count
    end
    if embed.subsampling < 0
        embed.subsampling = 1 / word_count
    end
    embed.distribution = distribution
    embed.vocabulary = collect(keys(distribution))
    @printf "corpus size: %d words\n" word_count
    @printf "Vocabulary size: %d\n" length(embed.vocabulary)

    initialize_embedding(embed, embed.init_type)        # initialize by the specified method
    initialize_network(embed, embed.network_type)

    # determine the position in the tree for every word
    for (w, code) in leaves_of(embed.classification_tree)
        embed.codebook[w] = code
    end

    function reduce_embeds!(embed, embs)
        n = length(embs)
        for word in embed.vocabulary
            embed.embedding[word] = sum(map(emb -> emb.embedding[word], embs)) / n
        end
        embed.trained_count = sum(map(emb->emb.trained_count, embs))
        embed
    end

    number_workers = nworkers()
    words_streams = parallel_words_of(corpus_filename, number_workers, subsampling = (embed.subsampling, embed.distribution))
    reduce_embeds!(embed, pmap(work_process, collect(repeated(embed, number_workers)), words_streams))
    embed
end

function initialize_embedding(embed :: WordEmbedding, randomly :: RandomInited)
    for i in embed.vocabulary
        embed.embedding[i] = rand(1, embed.dimension) * 2 - 1
    end
    embed
end

function initialize_network(embed :: WordEmbedding, huffman :: HuffmanTree)
    heap = PriorityQueue()
    for (word, freq) in embed.distribution
        node = BranchNode([], word, nothing)    # the data field of leaf node is its corresponding word.
        enqueue!(heap, node, freq)
    end
    while length(heap) > 1
        (node1, freq1) = peek(heap)
        dequeue!(heap)
        (node2, freq2) = peek(heap)
        dequeue!(heap)
        newnode = BranchNode([node1, node2], LinearClassifier(2, embed.dimension), nothing) # the data field of internal node is the classifier
        enqueue!(heap, newnode, freq1 + freq2)
    end
    embed.classification_tree = dequeue!(heap)
    embed
end

function initialize_network(embed :: WordEmbedding, ontology :: OntologyTree)
    function build_classifiers(node :: TreeNode)
        l = length(node.children)
        if l == 0
            return
        end
        node.data = LinearClassifier(l, embed.dimension)
        for c in node.children
            build_classifiers(c)
        end
    end
    embed.classification_tree = ontology.ontology
    embed
end

function Base.show(io :: IO, x :: WordEmbedding)
    @printf io "Word embedding(dimension = %d) of %d words, trained on %d words\n" x.dimension length(x.vocabulary) x.trained_count
    for (word, embed) in take(x.embedding, 5)
        @printf io "%s => %s\n" word string(embed)
    end
    if length(x.embedding) > 5
        @printf io "......"
    end
end

function find_nearest_words(embed :: WordEmbedding, word :: String; k = 1)
    if !haskey(embed.embedding, word)
        msg = @sprintf "'%s' doesn't present in the embedding\n" word
        warn(msg)
        return nothing
    end

    pq = PriorityQueue(Base.Order.Reverse)

    for (w, embed_w) in embed.embedding
        if w == word
            continue
        end
        enqueue!(pq, w, cosine_dist(vec(embed.embedding[word]), vec(embed_w)))
        if length(pq) > k
            dequeue!(pq)
        end
    end
    collect(pq)
end
