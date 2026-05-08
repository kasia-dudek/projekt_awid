# funkcje do klasyfikacji i liczenia straty

using LinearAlgebra

# operator logsumexp do stabilnego numerycznie sumowania w logarytmie
struct LogSumExpOp end

# operator straty cross-entropy liczony bezpośrednio z logits
struct LogitCrossEntropyOp end

# operator softmax - zamiana logits na prawdopodobieństwa
struct SoftmaxOp end

# stabilne numerycznie log(sum(exp(x)))
function logsumexp_forward(x)
    if ndims(x) == 1
        m = maximum(x)  # największa wartość do stabilizacji numerycznej
        return m + log(sum(exp.(x .- m)))  # logsumexp dla wektora
    elseif ndims(x) == 2
        m = maximum(x, dims=1)  # maksimum w każdej kolumnie batcha
        s = sum(exp.(x .- m), dims=1)  # suma exp po klasach
        return m .+ log.(s)  # logsumexp dla każdej próbki
    else
        error("logsumexp supports 1D or 2D logits")
    end
end

# tworzy w grafie węzeł operatora logsumexp
function logsumexp(x::GraphNode)
    return OperatorNode(LogSumExpOp(), x)
end

# forward logsumexp zwraca wartość operatora
forward(::OperatorNode{LogSumExpOp}, x) = logsumexp_forward(x)

# backward logsumexp zwraca gradient względem wejścia
backward(::OperatorNode{LogSumExpOp}, x, g) = (g .* exp.(x .- logsumexp_forward(x)),)

# liczy cross-entropy, czyli stratę dla klasyfikacji, bezpośrednio z logits i etykiet one-hot
function logitcrossentropy_forward(logits, labels)
    lse = logsumexp_forward(logits)  # log(sum(exp(logits)))
    if ndims(logits) == 1
        log_probs = logits .- lse  # log-softmax dla jednej próbki
        return -sum(labels .* log_probs)  # strata dla jednej próbki
    else
        log_probs = logits .- reshape(lse, 1, :)  # log-softmax dla batcha
        batch_size = size(logits, 2)  # liczba próbek w batchu
        return -sum(labels .* log_probs) / batch_size  # średnia strata po batchu
    end
end

# liczy softmax, czyli prawdopodobieństwa klas z logits
function softmax_forward(logits)
    if ndims(logits) == 1
        return exp.(logits .- logsumexp_forward(logits))  # softmax dla wektora
    elseif ndims(logits) == 2
        lse = logsumexp_forward(logits)  # logsumexp dla każdej próbki
        return exp.(logits .- reshape(lse, 1, :))  # softmax dla batcha
    else
        error("softmax supports 1D or 2D logits")
    end
end

# liczy gradient softmax bez budowania Jacobianu
function softmax_backward_from_probs(probs, g)
    if ndims(probs) == 1
        return probs .* (g .- sum(g .* probs))
    elseif ndims(probs) == 2
        return probs .* (g .- sum(g .* probs, dims=1))
    else
        error("softmax backward supports 1D or 2D tensors")
    end
end

# tworzy w grafie węzeł operatora softmax
function softmax(x::GraphNode)
    return OperatorNode(SoftmaxOp(), x)
end

# forward softmax zwraca prawdopodobieństwa klas
forward(::OperatorNode{SoftmaxOp}, x) = softmax_forward(x)

# backward softmax liczy gradient w wersji sfuzowanej (bez Jacobianu)
function backward(::OperatorNode{SoftmaxOp}, x, g)
    probs = softmax_forward(x)  # obliczone prawdopodobieństwa softmax
    return (softmax_backward_from_probs(probs, g),)
end

# tworzy w grafie węzeł straty cross-entropy
function logitcrossentropy(logits::GraphNode, labels)
    labels_node = labels isa GraphNode ? labels : Constant(labels)  # zamiana etykiet na węzeł grafu
    return OperatorNode(LogitCrossEntropyOp(), logits, labels_node)
end

# forward cross-entropy zwraca wartość straty
forward(::OperatorNode{LogitCrossEntropyOp}, logits, labels) = logitcrossentropy_forward(logits, labels)

# backward cross-entropy liczy gradient względem logits i ewentualnie labels
function backward(::OperatorNode{LogitCrossEntropyOp}, logits, labels, g)
    probs = softmax_forward(logits)  # prawdopodobieństwa z logits (bez osobnego węzła softmax)
    batch_size = ndims(logits) == 1 ? 1 : size(logits, 2)  # liczba próbek
    d_logits = (g / batch_size) .* (probs .- labels)  # gradient po logits

    if labels isa GraphNode
        d_labels = (g / batch_size) .* (-log.(probs))  # gradient po labels, jeśli są w grafie
        return (d_logits, d_labels)
    else
        return (d_logits, nothing)  # brak gradientu po stałych etykietach
    end
end