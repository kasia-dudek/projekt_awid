# CNN: proste Conv2D i MaxPool2D z poprawnym backward
# Przyjmujemy układ tensorów (H, W, C, N), czyli wysokość, szerokość, kanały, batch
using LinearAlgebra

# operator splotu 2D z informacją o paddingu i kroku
mutable struct Conv2DOp
    pad::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    x_padded
    out
    dx_padded
    dfilters
    dbias
    x_col
    y_col
    dy_col
    dx_col
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
    if pad_h > 0
        padded[1:pad_h, :, :, :] .= zero(eltype(x))
        padded[pad_h + H + 1:end, :, :, :] .= zero(eltype(x))
    end
    if pad_w > 0
        padded[:, 1:pad_w, :, :] .= zero(eltype(x))
        padded[:, pad_w + W + 1:end, :, :] .= zero(eltype(x))
    end
    padded[pad_h + 1:pad_h + H, pad_w + 1:pad_w + W, :, :] .= x  # wstawienie oryginalnych danych do środka
    return padded
end

# im2col dla pojedynczego elementu batcha (H, W, C)
function im2col_single!(x_col, x_padded_n, k_h, k_w, stride_h, stride_w, out_h, out_w, C_in)
    col = 1
    @inbounds for i in 1:out_h
        y = (i - 1) * stride_h + 1
        @inbounds for j in 1:out_w
            x0 = (j - 1) * stride_w + 1
            row = 1
            @inbounds for ic in 1:C_in
                @inbounds for p in 1:k_h
                    @inbounds for q in 1:k_w
                        x_col[row, col] = x_padded_n[y + p - 1, x0 + q - 1, ic]
                        row += 1
                    end
                end
            end
            col += 1
        end
    end
    return x_col
end

# zapisuje wynik GEMM do tensora wyjścia (H, W, C_out)
function write_y_col_to_out!(out_n, y_col, out_h, out_w, C_out)
    @inbounds for oc in 1:C_out
        col = 1
        @inbounds for i in 1:out_h
            @inbounds for j in 1:out_w
                out_n[i, j, oc] = y_col[oc, col]
                col += 1
            end
        end
    end
    return out_n
end

# odczytuje gradient wyjścia (H, W, C_out) do macierzy kolumnowej
function read_g_to_dy_col!(dy_col, g_n, out_h, out_w, C_out)
    @inbounds for oc in 1:C_out
        col = 1
        @inbounds for i in 1:out_h
            @inbounds for j in 1:out_w
                dy_col[oc, col] = g_n[i, j, oc]
                col += 1
            end
        end
    end
    return dy_col
end

# col2im dla pojedynczego elementu batcha
function col2im_single!(dx_padded_n, dx_col, k_h, k_w, stride_h, stride_w, out_h, out_w, C_in)
    col = 1
    @inbounds for i in 1:out_h
        y = (i - 1) * stride_h + 1
        @inbounds for j in 1:out_w
            x0 = (j - 1) * stride_w + 1
            row = 1
            @inbounds for ic in 1:C_in
                @inbounds for p in 1:k_h
                    @inbounds for q in 1:k_w
                        dx_padded_n[y + p - 1, x0 + q - 1, ic] += dx_col[row, col]
                        row += 1
                    end
                end
            end
            col += 1
        end
    end
    return dx_padded_n
end

# forward dla warstwy Conv2D
function conv2d_forward(op::Conv2DOp, x, filters, bias)
    T = eltype(x)
    pad_h, pad_w = op.pad  # padding w pionie i poziomie
    stride_h, stride_w = op.stride  # krok przesuwania filtra
    k_h, k_w, C_in, C_out = size(filters)  # rozmiar filtrów i liczba kanałów
    H, W, _, N = size(x)  # rozmiar wejścia
    padded_size = (H + 2 * pad_h, W + 2 * pad_w, C_in, N)
    if op.x_padded === nothing || size(op.x_padded) != padded_size
        op.x_padded = zeros(T, padded_size)  # bufor wejścia po paddingu
    end
    x_padded = pad_input!(op.x_padded, x, pad_h, pad_w)::Array{T,4}  # wejście po dodaniu paddingu
    #output_size = (input_size + 2*padding - kernel_size) / stride + 1
    out_h = (H + 2 * pad_h - k_h) ÷ stride_h + 1  # wysokość wyjścia
    out_w = (W + 2 * pad_w - k_w) ÷ stride_w + 1  # szerokość wyjścia
    out_size = (out_h, out_w, C_out, N)
    if op.out === nothing || size(op.out) != out_size
        op.out = zeros(T, out_size)  # bufor wyjścia
    end
    out = op.out::Array{T,4}
    K = k_h * k_w * C_in
    P = out_h * out_w
    if op.x_col === nothing || size(op.x_col) != (K, P)
        op.x_col = zeros(T, K, P)
    end
    if op.y_col === nothing || size(op.y_col) != (C_out, P)
        op.y_col = zeros(T, C_out, P)
    end
    x_col = op.x_col::Matrix{T}
    y_col = op.y_col::Matrix{T}
    Wk = reshape(filters, K, C_out)  # K x C_out

    @inbounds for n in 1:N
        x_padded_n = @view x_padded[:, :, :, n]
        im2col_single!(x_col, x_padded_n, k_h, k_w, stride_h, stride_w, out_h, out_w, C_in)
        mul!(y_col, transpose(Wk), x_col)  # (C_out x K) * (K x P) = (C_out x P)
        if bias !== nothing
            @inbounds for oc in 1:C_out
                b = bias[oc]
                @inbounds for p in 1:P
                    y_col[oc, p] += b
                end
            end
        end
        out_n = @view out[:, :, :, n]
        write_y_col_to_out!(out_n, y_col, out_h, out_w, C_out)
    end

    return out
end

# backward dla Conv2D: gradient po wejściu, filtrach i biasie
function conv2d_backward(node::OperatorNode{<:Conv2DOp}, x, filters, bias, g)
    T = eltype(x)
    pad_h, pad_w = node.f.pad  # odczytanie paddingu z operatora
    stride_h, stride_w = node.f.stride  # odczytanie kroku z operatora
    k_h, k_w, C_in, C_out = size(filters)  # wymiary filtrów
    H, W, _, N = size(x)  # wymiary wejścia
    padded_size = (H + 2 * pad_h, W + 2 * pad_w, C_in, N)
    if node.f.x_padded === nothing || size(node.f.x_padded) != padded_size
        node.f.x_padded = zeros(T, padded_size)
    end
    x_padded = pad_input!(node.f.x_padded, x, pad_h, pad_w)::Array{T,4}  # bufor wejścia po paddingu

    if node.f.dx_padded === nothing || size(node.f.dx_padded) != size(x_padded)
        node.f.dx_padded = zeros(T, size(x_padded))
    end
    dx_padded = node.f.dx_padded::Array{T,4}  # gradient po wejściu z paddingiem
    fill!(dx_padded, zero(T))

    if node.f.dfilters === nothing || size(node.f.dfilters) != size(filters)
        node.f.dfilters = zeros(T, size(filters))
    end
    dfilters = node.f.dfilters::Array{T,4}  # gradient po filtrach
    fill!(dfilters, zero(T))

    dbias = nothing
    if bias !== nothing
        if node.f.dbias === nothing || size(node.f.dbias) != size(bias)
            node.f.dbias = zeros(T, size(bias))
        end
        dbias = node.f.dbias::Array{T,1}
        fill!(dbias, zero(T))
    end
    out_h, out_w = size(g, 1), size(g, 2)  # rozmiar gradientu z kolejnej warstwy
    K = k_h * k_w * C_in
    P = out_h * out_w
    if node.f.x_col === nothing || size(node.f.x_col) != (K, P)
        node.f.x_col = zeros(T, K, P)
    end
    if node.f.dy_col === nothing || size(node.f.dy_col) != (C_out, P)
        node.f.dy_col = zeros(T, C_out, P)
    end
    if node.f.dx_col === nothing || size(node.f.dx_col) != (K, P)
        node.f.dx_col = zeros(T, K, P)
    end
    x_col = node.f.x_col::Matrix{T}
    dy_col = node.f.dy_col::Matrix{T}
    dx_col = node.f.dx_col::Matrix{T}

    Wk = reshape(filters, K, C_out)  # K x C_out
    dWk = reshape(dfilters, K, C_out)  # K x C_out

    @inbounds for n in 1:N
        x_padded_n = @view x_padded[:, :, :, n]
        g_n = @view g[:, :, :, n]
        dx_padded_n = @view dx_padded[:, :, :, n]

        im2col_single!(x_col, x_padded_n, k_h, k_w, stride_h, stride_w, out_h, out_w, C_in)
        read_g_to_dy_col!(dy_col, g_n, out_h, out_w, C_out)

        # dW += X_col * dY_col'
        mul!(dWk, x_col, transpose(dy_col), one(T), one(T))
        # dX_col = W * dY_col
        mul!(dx_col, Wk, dy_col)
        col2im_single!(dx_padded_n, dx_col, k_h, k_w, stride_h, stride_w, out_h, out_w, C_in)

        if dbias !== nothing
            @inbounds for oc in 1:C_out
                s = zero(T)
                @inbounds for p in 1:P
                    s += dy_col[oc, p]
                end
                dbias[oc] += s
            end
        end
    end

    dx = dx_padded[pad_h + 1:pad_h + H, pad_w + 1:pad_w + W, :, :]  # usunięcie paddingu z gradientu wejścia
    return dbias === nothing ? (dx, dfilters) : (dx, dfilters, dbias)
end

# tworzy węzeł Conv2D z biasem
function conv2d(x::GraphNode, filters::GraphNode, bias::GraphNode; pad=(0, 0), stride=(1, 1))
    return OperatorNode(Conv2DOp(pad, stride, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing), x, filters, bias)
end

# tworzy węzeł Conv2D bez biasu
function conv2d(x::GraphNode, filters::GraphNode; pad=(0, 0), stride=(1, 1))
    return OperatorNode(Conv2DOp(pad, stride, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing), x, filters)
end

# forward dla Conv2D z biasem
forward(node::OperatorNode{<:Conv2DOp}, x, filters, bias) = conv2d_forward(node.f, x, filters, bias)

# forward dla Conv2D bez biasu
forward(node::OperatorNode{<:Conv2DOp}, x, filters) = conv2d_forward(node.f, x, filters, nothing)

# backward dla Conv2D z biasem
backward(node::OperatorNode{<:Conv2DOp}, x, filters, bias, g) = conv2d_backward(node, x, filters, bias, g)

# backward dla Conv2D bez biasu
backward(node::OperatorNode{<:Conv2DOp}, x, filters, g) = conv2d_backward(node, x, filters, nothing, g)

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
    T = eltype(x)
    k_h, k_w = op.kernel  # rozmiar okna poolingu
    stride_h, stride_w = op.stride  # krok przesuwania okna
    H, W, C, N = size(x)  # rozmiar wejścia
    out_h = (H - k_h) ÷ stride_h + 1  # wysokość wyjścia
    out_w = (W - k_w) ÷ stride_w + 1  # szerokość wyjścia
    out_size = (out_h, out_w, C, N)
    if op.out === nothing || size(op.out) != out_size
        op.out = zeros(T, out_size)  # bufor wyjścia
    end
    out = op.out::Array{T,4}

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
function maxpool2d_backward(node::OperatorNode{<:MaxPool2DOp}, x, g)
    T = eltype(x)
    k_h, k_w = node.f.kernel  # rozmiar okna
    stride_h, stride_w = node.f.stride  # krok przesuwania
    H, W, C, N = size(x)  # rozmiar wejścia
    if node.f.dx === nothing || size(node.f.dx) != size(x)
        node.f.dx = zeros(T, size(x))  # bufor gradientu wejścia
    end
    dx = node.f.dx::Array{T,4}
    fill!(dx, zero(T))
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
forward(node::OperatorNode{<:MaxPool2DOp}, x) = maxpool2d_forward(node.f, x)

# backward dla maxpoolingu
backward(node::OperatorNode{<:MaxPool2DOp}, x, g) = maxpool2d_backward(node, x, g)

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