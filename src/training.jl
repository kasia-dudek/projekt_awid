# funkcje do treningu: zbieranie parametrów, tryb train/eval, SGD

# zbiera wszystkie parametry z modelu (rekurencyjnie po warstwach)
function params(model::Chain)
    ps = Variable[]
    for layer in model.layers
        append!(ps, params(layer))
    end
    return ps
end

# parametry warstwy Dense
function params(layer::Dense)
    return [layer.W, layer.b]
end

# parametry Conv2D (bias może nie istnieć)
function params(layer::Conv2D)
    return layer.bias === nothing ? [layer.filters] : [layer.filters, layer.bias]
end

# warstwy bez parametrów
function params(::Flatten)
    return Variable[]
end

function params(::MaxPool2D)
    return Variable[]
end

function params(::Dropout)
    return Variable[]
end

function params(::Function)
    return Variable[]
end

# pojedyncza zmienna jako parametr
function params(x::Variable)
    return [x]
end

# inne węzły nie mają parametrów
function params(::GraphNode)
    return Variable[]
end

# przełączenie modelu w tryb treningowy
function train!(model::Chain)
    for layer in model.layers
        train!(layer)
    end
end

# tryb ewaluacji (np. wyłączenie dropout)
function eval!(model::Chain)
    for layer in model.layers
        eval!(layer)
    end
end

# dropout w trybie treningowym
function train!(layer::Dropout)
    layer.training = true
end

# dropout w trybie testowym
function eval!(layer::Dropout)
    layer.training = false
end

# dla innych warstw nic nie robimy
function train!(_)
    return nothing
end

function eval!(_)
    return nothing
end

# zerowanie gradientów wszystkich parametrów
function zero_grad!(ps::AbstractVector{Variable})
    for p in ps
        zero_grad!(p)
    end
end

# krok SGD: aktualizacja wag
# parametr = parametr - learning_rate * gradient
# gradient: w którą stronę zmieniać parametr, żeby loss rósł najszybciej
# chcemy, żeby loss malał, więc idziemy w przeciwną stronę (- gradient)
function sgd_step!(ps::AbstractVector{Variable}, eta)
    for p in ps
        p.output .-= eta .* p.gradient # aktualizacja w kierunku ujemnym gradientu, eta to learning rate
    end
end