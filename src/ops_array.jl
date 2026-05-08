# operacje na tablicach: sum, mean, exp, log, max, mnożenie itd.

import Base: sum, exp, log, maximum, *
import Statistics: mean

# suma elementów
sum(x::GraphNode) = OperatorNode(sum, x)
forward(::OperatorNode{typeof(sum)}, x) = sum(x)
backward(::OperatorNode{typeof(sum)}, x, g) = (fill(g, size(x)),)

# średnia
mean(x::GraphNode) = OperatorNode(mean, x)
forward(::OperatorNode{typeof(mean)}, x) = mean(x)
backward(::OperatorNode{typeof(mean)}, x, g) = (fill(g / length(x), size(x)),)

# exp i log element-wise
exp(x::GraphNode) = OperatorNode(exp, x)
forward(::OperatorNode{typeof(exp)}, x) = exp.(x)
backward(::OperatorNode{typeof(exp)}, x, g) = (g .* exp.(x),)

log(x::GraphNode) = OperatorNode(log, x)
forward(::OperatorNode{typeof(log)}, x) = log.(x)
backward(::OperatorNode{typeof(log)}, x, g) = (g ./ x,)

# maksimum
maximum(x::GraphNode) = OperatorNode(maximum, x)
forward(::OperatorNode{typeof(maximum)}, x) = maximum(x)
backward(::OperatorNode{typeof(maximum)}, x, g) = (g .* (x .== maximum(x)),) # gradient tylko dla elementów równych maksimum

# backward dla mnożenia (różne przypadki)
function matmul_backward(x, y, g)
    if isa(x, Number) || isa(y, Number)
        return (g * y, g * x)
    elseif ndims(x) == 2 && ndims(y) == 1
        return (g * y', x' * g)
    elseif ndims(x) == 2 && ndims(y) == 2
        return (g * y', x' * g)
    elseif ndims(x) == 1 && ndims(y) == 1
        return (g * y, g * x)
    else
        return (g * y, x' * g)
    end
end

# przeciążenie *
*(x::GraphNode, y::GraphNode) = OperatorNode(*, x, y)
*(x::GraphNode, y::Number) = x * Constant(y)
*(x::Number, y::GraphNode) = Constant(x) * y

forward(::OperatorNode{typeof(*)}, x, y) = x * y
backward(::OperatorNode{typeof(*)}, x, y, g) = matmul_backward(x, y, g)

# ReLU
relu(x::GraphNode) = OperatorNode(relu, x)
forward(::OperatorNode{typeof(relu)}, x) = max.(0, x)
backward(::OperatorNode{typeof(relu)}, x, g) = (g .* (x .> 0),)

# dropout
mutable struct DropoutOp
    p::Float32
    mask::Union{Nothing, Array{Float32}}
    refresh::Bool
end

function dropout(x::GraphNode, p::Float32)
    return OperatorNode(DropoutOp(p, nothing, true), x)
end

function dropout(x::GraphNode, p::Real)
    return dropout(x, Float32(p))
end

forward(node::OperatorNode{DropoutOp}, x) = begin
    keep_scale = 1f0 / (1f0 - node.f.p)
    if node.f.mask === nothing || size(node.f.mask) != size(x)
        node.f.mask = zeros(Float32, size(x)...)
        node.f.refresh = true
    end
    if node.f.refresh
        rand!(node.f.mask)
        @. node.f.mask = ifelse(node.f.mask >= node.f.p, keep_scale, 0f0)
        node.f.refresh = false
    end
    return x .* node.f.mask
end

backward(node::OperatorNode{DropoutOp}, x, g) = (g .* node.f.mask,)

function refresh_dropout_mask!(op::DropoutOp)
    op.refresh = true
    return nothing
end

# flatten
flatten(x::AbstractArray) = ndims(x) == 4 ? reshape(x, prod(size(x)[1:3]), size(x, 4)) : reshape(x, :) 
# spłaszczanie do 2D (HWN -> (HWN)x1) lub do 1D (dla innych wymiarów)
flatten(x::GraphNode) = OperatorNode(flatten, x)
forward(::OperatorNode{typeof(flatten)}, x) = flatten(x)
backward(::OperatorNode{typeof(flatten)}, x, g) = (reshape(g, size(x)),)