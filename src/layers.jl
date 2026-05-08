# warstwa Dense (czyli klasyczna warstwa w pełni połączona)
struct Dense
    W  # macierz wag
    b  # bias
end

# inicjalizacja wag metodą He (dobra przy ReLU)
# metoda He: losowe wartości z rozkładu normalnego, skalowane przez sqrt(2 / liczba_wejść)
function Dense(in_dim::Int, out_dim::Int)
    W = Variable(randn(Float32, out_dim, in_dim) .* sqrt(2f0 / in_dim), name=:W)  # losowe wagi
    b = Variable(zeros(Float32, out_dim), name=:b)  # bias na start = 0
    return Dense(W, b)
end

# forward Dense: Wx + b
function (layer::Dense)(x)
    return layer.W * x + layer.b  # klasyczne przekształcenie liniowe
end

# flatten - spłaszcza tensor (np. z CNN) do wektora
struct Flatten end

function (f::Flatten)(x)
    return flatten(x)  # zamiana np. (H,W,C,N) -> (features, N)
end

# Dropout - losowe wyłączanie neuronów w trakcie treningu
mutable struct Dropout
    p::Float32      # prawdopodobieństwo wyzerowania neuronu
    training::Bool  # czy jesteśmy w trybie treningowym
end

# konstruktor z Float32
function Dropout(p::Float32)
    return Dropout(p, true)
end

# konstruktor przyjmujący dowolny typ liczbowy
function Dropout(p::Real)
    return Dropout(Float32(p), true)
end

# forward Dropout
function (layer::Dropout)(x)
    if layer.training
        return dropout(x, layer.p)  # losowo zeruje część wartości
    else
        return x  # w testowaniu dropout jest wyłączony
    end
end

# Chain - kontener na kolejne warstwy (jak Sequential w PyTorch)
struct Chain
    layers  # lista warstw
end

# tworzy chain z dowolnej liczby warstw
function Chain(layers...)
    return Chain(collect(layers))
end

# forward przez cały model (przechodzimy warstwa po warstwie)
# (c::Chain)(x) - wywołanie modelu jak funkcji, np. model(x)
function (c::Chain)(x)
    for layer in c.layers
        x = layer(x)  # każda warstwa przetwarza wynik poprzedniej
    end
    return x
end