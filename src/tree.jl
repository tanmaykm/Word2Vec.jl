abstract TreeNode

type BranchNode <: TreeNode
    children :: Array{BranchNode, 1}
    data
    extrainfo
end

type NullNode <: TreeNode
end

const nullnode = NullNode()

function leaves_of(root::TreeNode)
    code = Int64[]
    function traverse(node::TreeNode)
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

function internal_nodes_of(root::TreeNode)
    function traverse(node::TreeNode)
        if node == nullnode
            return
        end
        if length(node.children) != 0
            produce(node)
        end
        for child in node.children
            traverse(child)
        end
    end
    Task(() -> traverse(root))
end

function average_height(tree::TreeNode)
    (h, c) = (0, 0)
    for (_, path) in leaves_of(tree)
        h += length(path)
        c += 1
    end
    h / c
end
