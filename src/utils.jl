function words_of(fp::IOStream; subsampling=(0, nothing), startpoint=-1, endpoint=-1)
    (startpoint >= 0) && seek(fp, startpoint)
    (rate, distr) = subsampling
    function producer()
        out = IOBuffer()
        while !eof(fp)
            (endpoint >= 0) && (position(fp) > endpoint) && break
            c = read(fp, Char)
            if c == ' ' || c == '\n'
                s = takebuf_string(out)
                if !isempty(s)
                    if rate > 0 && haskey(distr, s)
                        rs = distr[s]
                        prob = (sqrt(rs / rate) + 1) * rate / rs
                        if(prob < rand())
                            # @printf "throw %s, prob is %f\n" s prob
                            continue
                        end
                    end
                    produce(s)
                end
            else
                write(out, c)
            end
        end
    end
    return Task(producer)
end

function words_of(filename::AbstractString; subsampling=(0, nothing), startpoint = -1, endpoint = -1)
    function wrapper()
        fp = open(filename, "r")
        t = words_of(fp, subsampling = subsampling, startpoint=startpoint, endpoint=endpoint)
        while !istaskdone(t)
            res = consume(t)
            if res == () && istaskdone(t)
                break
            end
            produce(res)
        end
        close(fp)
    end
    Task(wrapper)
end


function parallel_words_of(filename::AbstractString, num_workers::Integer; subsampling=(0, nothing))
    #fp = open(filename, "r")
    #seekend(fp)
    #flen = position(fp)
    #close(fp)
    flen = filesize(filename)

    per_len = floor(Int, flen / num_workers)
    cursor = 0
    res = Array(Any, num_workers)
    for i in 1:num_workers
        last = (i == num_workers ? flen - 1 : cursor + per_len - 1)
        res[i] = words_of(filename, subsampling=subsampling, startpoint=cursor, endpoint=last)
        cursor += per_len
    end
    res
end

function sliding_window(words; lsize = 5, rsize = 5)
    size = lsize + 1 + rsize

    function producer()
        # initialize the window
        window = collect(take(words, size))
        if length(window) != size
            return
        end
        produce(window)

        # move the window (notice that we don't need to drop the first window-size items
        # of the words iterator because it is a producer-consumer Task iterator, not a 
        # stream-like functional iterator)
        for w in words
            shift!(window)
            push!(window, w)
            produce(window)
       end
    end
    return Task(producer)
end

abstract TreeNode
type BranchNode <: TreeNode
    children :: Array{BranchNode, 1}
    data
    extrainfo
end
type NullNode <: TreeNode
end
const nullnode = NullNode()

function leaves_of(root :: TreeNode)
    code = Int64[]
    function traverse(node :: TreeNode)
        if node == nullnode
            return
        end
        if length(node.children) == 0
            produce((node.data, copy(code)))    # notice that we should copy the current state of code
        end
        for (index, child) in enumerate(node.children)
            push!(code, index)
            traverse(child)
            pop!(code)
        end
    end
    Task(() -> traverse(root))
end

# partition an array to n parts
function partition{T}(a::Array{T}, n::Integer)
    b = Array{T}[]
    t = floor(Int, length(a) / n)
    cursor = 1
    for i in 1:n
        push!(b, a[cursor : (i == n ? length(a) : cursor + t - 1)])
        cursor += t
    end
    b
end
