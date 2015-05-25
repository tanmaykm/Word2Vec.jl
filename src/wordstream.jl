type WordStream
    fp::Union(IO, AbstractString)
    startpoint::Int64
    endpoint::Int64
    buffer::IOBuffer

    # filter configuration
    rate::Float64   # if rate > 0, words will be subsampled according to distr
    filter::Bool    # if filter is true, only words present in the keys(distr) will be considered
    distr::Dict{AbstractString, Float64}
end

function words_of(file::Union(IO,AbstractString); subsampling=(0,false,nothing), startpoint=-1, endpoint=-1)
    rate, filter, distr = subsampling
    WordStream(file, startpoint, endpoint, IOBuffer(), rate, filter, (rate==0 && !filter) ? Dict{AbstractString,Float64}() : distr)
end

function parallel_words_of(filename::AbstractString, num_workers::Integer; subsampling=(0,false,nothing))
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

function Base.start(ws::WordStream)
    if isa(ws.fp, AbstractString)
        ws.fp = open(ws.fp)
    end
    if ws.startpoint >= 0
        seek(ws.fp, ws.startpoint)
    else
        ws.startpoint = 0
        ws.endpoint = filesize(ws.fp)
        #seekend(ws.fp)
        #ws.endpoint = position(ws.fp)
        #seekstart(ws.fp)
    end
    nothing
end

function Base.done(ws::WordStream, state)
    while !eof(ws.fp)
        if ws.endpoint >= 0 && position(ws.fp) > ws.endpoint
            break
        end
        c = read(ws.fp, Char)
        if c == ' ' || c == '\n' || c == '\0' || c == '\r'
            s = takebuf_string(ws.buffer)
            if s == "" || (ws.filter && !haskey(ws.distr, s))
                continue
            end
            if ws.rate > 0
                prob = (sqrt(ws.distr[s] / ws.rate) + 1) * ws.rate / ws.distr[s]
                if(prob < rand())
                    # @printf "throw %s, prob is %f\n" s prob
                    continue;
                end
            end
            write(ws.buffer, s)
            return false
        else
            write(ws.buffer, c)
        end
    end
    #close(ws.fp)
    return true
end

function Base.next(ws::WordStream, state)
    (takebuf_string(ws.buffer), nothing)
end



type SlidingWindow
    ws::WordStream
    lsize::Int64
    rsize::Int64
end

function Base.start(window::SlidingWindow)
    convert(Array{AbstractString,1}, collect(take(window.ws, window.lsize + 1 + window.rsize)))
end

function Base.done(window::SlidingWindow, w::Array{AbstractString})
    done(window.ws, nothing)
end

function Base.next(window::SlidingWindow, w::Array{AbstractString})
    shift!(w)
    push!(w, next(window.ws, nothing)[1])
    (w, w)
end

function sliding_window(words; lsize=5, rsize=5)
    SlidingWindow(words, lsize, rsize)
end
