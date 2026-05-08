# zapisanie obliczeń jako grafu
# policzenie wartości od początku do końca — forward
# policzenie gradientów od końca do początku — backward
# sortowanie topologiczne

# wspólny typ bazowy dla wszystkich węzłów w grafie
abstract type GraphNode end

# stała w grafie, ma tylko wartość i nie liczymy dla niej gradientu
struct Constant{T} <: GraphNode
    output::T
end

# zmienna w grafie, przechowuje wartość, gradient i nazwę
mutable struct Variable{T} <: GraphNode
    output::T
    gradient::T
    name::Symbol
end

# węzeł operatora, przechowuje funkcję, wejścia, wynik i gradient
mutable struct OperatorNode{F} <: GraphNode
    f::F
    inputs::Tuple
    output
    gradient
end

# Konstruktory

# tworzy stałą na podstawie podanej wartości
Constant(x) = Constant{typeof(x)}(x)

# tworzy zmienną i od razu inicjalizuje jej gradient zerami
function Variable(x; name::Symbol=:var)
    g = x isa Number ? zero(x) : zeros(eltype(x), size(x)...)
    return Variable{typeof(x)}(x, g, name)
end

# tworzy węzeł operatora z funkcją i wejściami
function OperatorNode(f, inputs...)
    return OperatorNode{typeof(f)}(f, inputs, nothing, nothing)
end

# Dostęp do wartości / gradientu

# odczyt wartości stałej
value(x::Constant) = x.output

# odczyt wartości zmiennej
value(x::Variable) = x.output

# odczyt wartości operatora
value(x::OperatorNode) = x.output

# odczyt gradientu zmiennej
grad(x::Variable) = x.gradient

# odczyt gradientu operatora
grad(x::OperatorNode) = x.gradient

# Reset gradientów

# dla stałej nic nie trzeba resetować
reset!(x::Constant) = nothing

# zeruje gradient zmiennej
function reset!(x::Variable)
    if x.output isa Number
        x.gradient = zero(x.output)  # zerowanie dla skalaru
    else
        fill!(x.gradient, zero(eltype(x.output)))  # zerowanie dla tablicy
    end
    return nothing
end

# dla operatora kasujemy zapamiętany gradient
reset!(x::OperatorNode) = (x.gradient = nothing)

# Topological sort

# zwraca węzły w kolejności potrzebnej do forward/backward
function topo_sort(head::GraphNode)
    visited = Set{GraphNode}()  # zbiór odwiedzonych węzłów
    order = GraphNode[]  # końcowa kolejność przechodzenia po grafie

    function visit(node::GraphNode)
        if node in visited
            return  # jeśli już odwiedzony, to nic nie robimy
        end

        push!(visited, node)  # oznaczenie węzła jako odwiedzonego

        if node isa OperatorNode
            for input in node.inputs
                visit(input)  # najpierw schodzimy do wejść
            end
        end

        push!(order, node)  # dopiero potem dodajemy węzeł do kolejności
    end

    visit(head)
    return order
end

# Compute / forward pass

# dla stałej nie trzeba nic liczyć
compute!(x::Constant) = nothing

# dla zmiennej też nie trzeba nic liczyć, bo wartość już istnieje
compute!(x::Variable) = nothing

# dla operatora liczymy output na podstawie wejść
function compute!(node::OperatorNode)
    xs = ntuple(i -> value(node.inputs[i]), length(node.inputs))  # pobranie wartości wejść
    node.output = forward(node, xs...)  # wywołanie odpowiedniego forward
    return nothing
end

# przejście forward po wszystkich węzłach w dobrej kolejności
function forward!(order::Vector{GraphNode})
    for node in order
        compute!(node)  # obliczenie wartości danego węzła

        if node isa OperatorNode
            node.gradient = nothing  # gradient operatora będzie liczony później
        elseif node isa Variable
            if node.output isa Number
                node.gradient = zero(node.output)  # zerowanie gradientu skalaru
            else
                fill!(node.gradient, zero(eltype(node.output)))  # zerowanie gradientu tablicy
            end
        end
    end

    return value(last(order))  # zwracamy wynik końcowego węzła
end

# Akumulacja gradientów

# dla stałej gradientu nie zbieramy
function accumulate!(node::Constant, g)
    return nothing
end

# dopisuje gradient do zmiennej
function accumulate!(node::Variable, g)
    if g === nothing
        return nothing  # jeśli gradient nie istnieje, to nic nie robimy
    end

    if node.output isa Number
        node.gradient += g  # akumulacja gradientu dla skalaru
    else
        node.gradient .+= g  # akumulacja gradientu element po elemencie
    end

    return nothing
end

# dopisuje gradient do operatora
function accumulate!(node::OperatorNode, g)
    if g === nothing
        return nothing  # jeśli brak gradientu, to nic nie robimy
    end

    if isnothing(node.gradient)
        if g isa Number
            node.gradient = g  # pierwszy gradient dla skalaru
        else
            node.gradient = copy(g)  # pierwszy gradient dla tablicy
        end
    else
        if node.gradient isa Number
            node.gradient += g  # kolejne gradienty sumujemy
        else
            node.gradient .+= g  # sumowanie gradientów dla tablicy
        end
    end

    return nothing
end

# Backward pass

# przejście backward po grafie w odwrotnej kolejności
function backward!(order::Vector{GraphNode}; seed=1f0)
    result = last(order)  # końcowy węzeł grafu

    if value(result) isa Number
        result.gradient = seed  # gradient startowy dla skalaru
    else
        result.gradient = fill(seed, size(value(result)))  # gradient startowy dla tensora
    end

    for node in reverse(order)
        if node isa OperatorNode
            xs = ntuple(i -> value(node.inputs[i]), length(node.inputs))  # pobranie wartości wejść
            grads = backward(node, xs..., node.gradient)  # policzenie gradientów po wejściach

            for (input, g) in zip(node.inputs, grads)
                accumulate!(input, g)  # przekazanie gradientu do poprzednich węzłów
            end
        end
    end

    return nothing
end

# Zerowanie gradientów

# zeruje gradient jednej zmiennej
function zero_grad!(x::Variable)
    if x.output isa Number
        x.gradient = zero(x.output)  # zerowanie dla skalaru
    else
        fill!(x.gradient, zero(eltype(x.output)))  # zerowanie dla tablicy
    end
    return nothing
end

# zeruje gradienty wszystkich parametrów z wektora
function zero_grad!(ps::AbstractVector{Variable})
    for p in ps
        zero_grad!(p)
    end
    return nothing
end

# zeruje gradienty dla całego grafu
function zero_grad!(order::Vector{GraphNode})
    for node in order
        reset!(node)
    end
    return nothing
end