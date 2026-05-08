# CNN: proste Conv2D i MaxPool2D z poprawnym backward
# Przyjmujemy układ tensorów (H, W, C, N), czyli wysokość, szerokość, kanały, batch

# operator splotu 2D z informacją o paddingu i kroku
mutable struct Conv2DOp
    pad::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    x_padded
    out
    dx_padded
    dfilters
    dbias
end

# operator maxpoolingu 2D z rozmiarem okna i krokiem
mutable struct MaxPool2DOp
    kernel::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    out
    dx
end

# dodaje zerowy padding wokół wejścia
function pad_input!(padded, x, pad_h, pad_w)
    H, W, C, N = size(x)  # wymiary wejścia
    fill!(padded, zero(eltype(x)))
    padded[pad_h + 1:pad_h + H, pad_w + 1:pad_w + W, :, :] .= x  # wstawienie oryginalnych danych do środka
    return padded
end

# forward dla warstwy Conv2D
function conv2d_forward(op::Conv2DOp, x, filters, bias)
    pad_h, pad_w = op.pad  # padding w pionie i poziomie
    stride_h, stride_w = op.stride  # krok przesuwania filtra
    k_h, k_w, C_in, C_out = size(filters)  # rozmiar filtrów i liczba kanałów
    H, W, _, N = size(x)  # rozmiar wejścia
    padded_size = (H + 2 * pad_h, W + 2 * pad_w, C_in, N)
    if op.x_padded === nothing || size(op.x_padded) != padded_size || eltype(op.x_padded) != eltype(x)
        op.x_padded = zeros(eltype(x), padded_size)
    end
    x_padded = pad_input!(op.x_padded, x, pad_h, pad_w)  # wejście po dodaniu paddingu
    #output_size = (input_size + 2*padding - kernel_size) / stride + 1
    out_h = (H + 2 * pad_h - k_h) ÷ stride_h + 1  # wysokość wyjścia
    out_w = (W + 2 * pad_w - k_w) ÷ stride_w + 1  # szerokość wyjścia
    out_size = (out_h, out_w, C_out, N)
    if op.out === nothing || size(op.out) != out_size || eltype(op.out) != eltype(x)
        op.out = zeros(eltype(x), out_size)
    end
    out = op.out
    fill!(out, zero(eltype(x)))

    @inbounds for n in 1:N # który obrazek w batchu
        @inbounds for oc in 1:C_out # który kanał wyjściowy (filtr)
            @inbounds for i in 1:out_h # która pozycja w pionie
                @inbounds for j in 1:out_w # która pozycja w poziomie
                    y = (i - 1) * stride_h + 1  # początek okna w pionie
                    x0 = (j - 1) * stride_w + 1  # początek okna w poziomie
                    acc = zero(eltype(x))  # akumulator sumy dla jednego pola wyjścia
                    @inbounds for ic in 1:C_in
                        @inbounds for p in 1:k_h
                            @inbounds for q in 1:k_w
                                acc += filters[p, q, ic, oc] * x_padded[y + p - 1, x0 + q - 1, ic, n]  # splot, konkretna waga filtra razy konkretny pixel
                            end
                        end
                    end
                    out[i, j, oc, n] = acc  # zapis wyniku splotu
                    if bias !== nothing
                        out[i, j, oc, n] += bias[oc]  # dodanie biasu dla kanału wyjściowego
                    end
                end
            end
        end
    end
    return out
end

# backward dla Conv2D: gradient po wejściu, filtrach i biasie
function conv2d_backward(node::OperatorNode{Conv2DOp}, x, filters, bias, g)
    pad_h, pad_w = node.f.pad  # odczytanie paddingu z operatora
    stride_h, stride_w = node.f.stride  # odczytanie kroku z operatora
    k_h, k_w, C_in, C_out = size(filters)  # wymiary filtrów
    H, W, _, N = size(x)  # wymiary wejścia
    x_padded = node.f.x_padded  # wejście z paddingiem zapamiętane z forward

    if node.f.dx_padded === nothing || size(node.f.dx_padded) != size(x_padded) || eltype(node.f.dx_padded) != eltype(x)
        node.f.dx_padded = zeros(eltype(x), size(x_padded))
    end
    dx_padded = node.f.dx_padded  # gradient po wejściu z paddingiem
    fill!(dx_padded, zero(eltype(x)))

    if node.f.dfilters === nothing || size(node.f.dfilters) != size(filters) || eltype(node.f.dfilters) != eltype(filters)
        node.f.dfilters = zeros(eltype(filters), size(filters))
    end
    dfilters = node.f.dfilters  # gradient po filtrach
    fill!(dfilters, zero(eltype(filters)))

    dbias = nothing
    if bias !== nothing
        if node.f.dbias === nothing || size(node.f.dbias) != size(bias) || eltype(node.f.dbias) != eltype(bias)
            node.f.dbias = zeros(eltype(bias), size(bias))
        end
        dbias = node.f.dbias
        fill!(dbias, zero(eltype(bias)))
    end
    out_h, out_w = size(g, 1), size(g, 2)  # rozmiar gradientu z kolejnej warstwy

    @inbounds for n in 1:N
        @inbounds for oc in 1:C_out
            @inbounds for i in 1:out_h
                @inbounds for j in 1:out_w
                    y = (i - 1) * stride_h + 1  # początek okna w pionie
                    x0 = (j - 1) * stride_w + 1  # początek okna w poziomie
                    grad = g[i, j, oc, n]  # lokalny gradient dla danej pozycji wyjścia
                    @inbounds for ic in 1:C_in
                        @inbounds for p in 1:k_h
                            @inbounds for q in 1:k_w
                                dfilters[p, q, ic, oc] += grad * x_padded[y + p - 1, x0 + q - 1, ic, n]  # gradient po wagach
                                dx_padded[y + p - 1, x0 + q - 1, ic, n] += grad * filters[p, q, ic, oc]  # gradient po wejściu
                            end
                        end
                    end
                    if dbias !== nothing
                        dbias[oc] += grad  # bias zbiera sumę gradientów dla danego kanału
                    end
                end
            end
        end
    end

    dx = dx_padded[pad_h + 1:pad_h + H, pad_w + 1:pad_w + W, :, :]  # usunięcie paddingu z gradientu wejścia
    return dbias === nothing ? (dx, dfilters) : (dx, dfilters, dbias)
end

# tworzy węzeł Conv2D z biasem
function conv2d(x::GraphNode, filters::GraphNode, bias::GraphNode; pad=(0, 0), stride=(1, 1))
    return OperatorNode(Conv2DOp(pad, stride, nothing, nothing, nothing, nothing, nothing), x, filters, bias)
end

# tworzy węzeł Conv2D bez biasu
function conv2d(x::GraphNode, filters::GraphNode; pad=(0, 0), stride=(1, 1))
    return OperatorNode(Conv2DOp(pad, stride, nothing, nothing, nothing, nothing, nothing), x, filters)
end

# forward dla Conv2D z biasem
forward(node::OperatorNode{Conv2DOp}, x, filters, bias) = conv2d_forward(node.f, x, filters, bias)

# forward dla Conv2D bez biasu
forward(node::OperatorNode{Conv2DOp}, x, filters) = conv2d_forward(node.f, x, filters, nothing)

# backward dla Conv2D z biasem
backward(node::OperatorNode{Conv2DOp}, x, filters, bias, g) = conv2d_backward(node, x, filters, bias, g)

# backward dla Conv2D bez biasu
backward(node::OperatorNode{Conv2DOp}, x, filters, g) = conv2d_backward(node, x, filters, nothing, g)

# warstwa Conv2D przechowująca parametry i ustawienia
struct Conv2D
    filters::Variable
    bias::Union{Variable, Nothing}
    pad::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

# konstruktor warstwy Conv2D z inicjalizacją He
function Conv2D(in_ch::Int, out_ch::Int, kernel::Tuple{Int,Int}; stride=1, padding=0, bias=true)
    stride_tuple = (stride, stride)  # zamiana na parę liczb
    pad_tuple = (padding, padding)  # zamiana na parę liczb
    # inicjalizacja He dobra np. dla ReLU
    fan_in = kernel[1] * kernel[2] * in_ch  # liczba wejść do jednego neuronu
    filters = Variable(randn(Float32, kernel[1], kernel[2], in_ch, out_ch) .* sqrt(2f0 / fan_in), name=:W)  # wagi filtrów
    bias_var = bias ? Variable(zeros(Float32, out_ch), name=:b) : nothing  # bias albo brak biasu
    return Conv2D(filters, bias_var, pad_tuple, stride_tuple)
end

# wywołanie warstwy Conv2D na wejściu
function (layer::Conv2D)(x)
    if layer.bias === nothing
        return conv2d(x, layer.filters; pad=layer.pad, stride=layer.stride)  # splot bez biasu
    else
        return conv2d(x, layer.filters, layer.bias; pad=layer.pad, stride=layer.stride)  # splot z biasem
    end
end

# forward dla maxpoolingu 2D
# @inbounds - „nie sprawdzaj, czy indeks tablicy jest poprawny”
function maxpool2d_forward(op::MaxPool2DOp, x)
    k_h, k_w = op.kernel  # rozmiar okna poolingu
    stride_h, stride_w = op.stride  # krok przesuwania okna
    H, W, C, N = size(x)  # rozmiar wejścia
    out_h = (H - k_h) ÷ stride_h + 1  # wysokość wyjścia
    out_w = (W - k_w) ÷ stride_w + 1  # szerokość wyjścia
    out_size = (out_h, out_w, C, N)
    if op.out === nothing || size(op.out) != out_size || eltype(op.out) != eltype(x)
        op.out = zeros(eltype(x), out_size)
    end
    out = op.out  # tensor wyjściowy
    fill!(out, zero(eltype(x)))

    @inbounds for n in 1:N
        @inbounds for c in 1:C
            @inbounds for i in 1:out_h
                @inbounds for j in 1:out_w
                    y = (i - 1) * stride_h + 1  # początek okna w pionie
                    x0 = (j - 1) * stride_w + 1  # początek okna w poziomie
                    max_val = typemin(eltype(x))  # startowa wartość minimum typu
                    @inbounds for p in 1:k_h
                        @inbounds for q in 1:k_w
                            max_val = max(max_val, x[y + p - 1, x0 + q - 1, c, n])  # szukanie maksimum w oknie
                        end
                    end
                    out[i, j, c, n] = max_val  # zapis maksymalnej wartości
                end
            end
        end
    end
    return out
end

# backward dla maxpoolingu rozdziela gradient do elementów maksymalnych
function maxpool2d_backward(node::OperatorNode{MaxPool2DOp}, x, g)
    k_h, k_w = node.f.kernel  # rozmiar okna
    stride_h, stride_w = node.f.stride  # krok przesuwania
    H, W, C, N = size(x)  # rozmiar wejścia
    if node.f.dx === nothing || size(node.f.dx) != size(x) || eltype(node.f.dx) != eltype(x)
        node.f.dx = zeros(eltype(x), size(x))
    end
    dx = node.f.dx  # gradient po wejściu
    fill!(dx, zero(eltype(x)))
    out_h, out_w = size(g, 1), size(g, 2)  # rozmiar gradientu wyjściowego

    @inbounds for n in 1:N
        @inbounds for c in 1:C
            @inbounds for i in 1:out_h
                @inbounds for j in 1:out_w
                    y = (i - 1) * stride_h + 1  # początek okna w pionie
                    x0 = (j - 1) * stride_w + 1  # początek okna w poziomie
                    max_val = typemin(eltype(x))  # startowa wartość maksimum
                    @inbounds for p in 1:k_h
                        @inbounds for q in 1:k_w
                            max_val = max(max_val, x[y + p - 1, x0 + q - 1, c, n])  # znalezienie maksimum w oknie
                        end
                    end
                    count = 0  # liczba elementów równych maksimum
                    @inbounds for p in 1:k_h
                        @inbounds for q in 1:k_w
                            if x[y + p - 1, x0 + q - 1, c, n] == max_val
                                count += 1  # zliczamy ile razy maksimum wystąpiło
                            end
                        end
                    end
                    grad = g[i, j, c, n] / count  # dzielimy gradient po wszystkich maksimach
                    @inbounds for p in 1:k_h
                        @inbounds for q in 1:k_w
                            if x[y + p - 1, x0 + q - 1, c, n] == max_val
                                dx[y + p - 1, x0 + q - 1, c, n] += grad  # przekazanie gradientu tylko do maksimów
                            end
                        end
                    end
                end
            end
        end
    end
    return (dx,)
end

# tworzy węzeł operatora maxpoolingu
function maxpool2d(x::GraphNode, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int})
    return OperatorNode(MaxPool2DOp(kernel, stride, nothing, nothing), x)
end

# forward dla maxpoolingu
forward(node::OperatorNode{MaxPool2DOp}, x) = maxpool2d_forward(node.f, x)

# backward dla maxpoolingu
backward(node::OperatorNode{MaxPool2DOp}, x, g) = maxpool2d_backward(node, x, g)

# warstwa MaxPool2D przechowująca parametry poolingu
struct MaxPool2D
    kernel::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

# konstruktor warstwy MaxPool2D
function MaxPool2D(pool_size::Tuple{Int,Int}; stride::Tuple{Int,Int}=pool_size)
    return MaxPool2D(pool_size, stride)
end

# wywołanie warstwy MaxPool2D na wejściu
function (layer::MaxPool2D)(x)
    return maxpool2d(x, layer.kernel, layer.stride)
end