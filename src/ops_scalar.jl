# podstawowe operacje: +, -, /

import Base: +, -, /

# tworzenie węzłów grafu dla operacji
+(x::GraphNode, y::GraphNode) = OperatorNode(+, x, y)
-(x::GraphNode, y::GraphNode) = OperatorNode(-, x, y)
/(x::GraphNode, y::GraphNode) = OperatorNode(/, x, y)

# obsługa przypadków mieszanych (GraphNode + liczba)
+(x::GraphNode, y::Number) = x + Constant(y)
+(x::Number, y::GraphNode) = Constant(x) + y

-(x::GraphNode, y::Number) = x - Constant(y)
-(x::Number, y::GraphNode) = Constant(x) - y

/(x::GraphNode, y::Number) = x / Constant(y)
/(x::Number, y::GraphNode) = Constant(x) / y

# dopasowanie gradientu do kształtu (np. przy broadcastingu)
# broadcasting: jeśli x i y mają różne kształty, to gradient g może mieć większy kształt niż y, 
# więc musimy go zredukować do kształtu y
function reduce_grad(g, shape)
    if size(g) == shape
        return g
    end
    nd = ndims(g)
    target = ntuple(i -> i <= length(shape) ? shape[i] : 1, nd)
    out = g
    for d = nd:-1:1
        if target[d] == 1
            out = sum(out; dims=d)  # sumowanie po wymiarach broadcastowanych
        end
    end
    out = dropdims(out; dims=Tuple(i for i in 1:nd if target[i] == 1))
    return reshape(out, shape)
end

# forward
forward(::OperatorNode{typeof(+)}, x, y) = x .+ y
forward(::OperatorNode{typeof(-)}, x, y) = x .- y
forward(::OperatorNode{typeof(/)}, x, y) = x ./ y

# backward (z uwzględnieniem broadcastingu)
backward(::OperatorNode{typeof(+)}, x, y, g) = (g, reduce_grad(g, size(y)))
backward(::OperatorNode{typeof(-)}, x, y, g) = (g, reduce_grad(-g, size(y)))
backward(::OperatorNode{typeof(/)}, x, y, g) = (g ./ y, reduce_grad(-g .* x ./ (y .^ 2), size(y))) 
