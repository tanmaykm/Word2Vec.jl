# serialize to a file
function save(item, filename::AbstractString)
    open(filename, "w") do fp
        save(item, fp)
    end
end
save(item, fp::IO) = serialize(fp, item)

# restore from a file
function restore(filename::AbstractString)
    open(filename, "r") do fp
        restore(fp)
    end
end
restore(fp::IO) = deserialize(fp)

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
