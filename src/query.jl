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
