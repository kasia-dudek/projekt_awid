using Downloads
using CodecZlib
using Random

# link do datasetu FashionMNIST
const FASHION_MNIST_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

# tworzy katalog jeśli nie istnieje
function ensure_dir(dir)
    if !isdir(dir)
        mkpath(dir)  # tworzenie folderu
    end
    return abspath(dir)  # zwraca absolutną ścieżkę
end

# odczyt liczby UInt32 w big-endian (format danych MNIST)
function read_uint32_be(io)
    return ntoh(read(io, UInt32))  # zamiana na poprawny porządek bajtów
end

# wczytuje obrazy z pliku .gz
function read_images(path)
    io = GzipDecompressorStream(open(path, "r"))  # otwarcie i dekompresja pliku
    magic = read_uint32_be(io)
    @assert magic == 2051 "Invalid image file"  # sprawdzenie typu pliku
    n = Int(read_uint32_be(io))  # liczba obrazów
    rows = Int(read_uint32_be(io))  # wysokość
    cols = Int(read_uint32_be(io))  # szerokość
    raw = read(io, n * rows * cols)  # surowe dane pikseli
    close(io)
    data = Float32.(raw) ./ 255f0  # normalizacja do [0,1]
    images = reshape(data, cols, rows, n)  # reshaping danych
    images = permutedims(images, (2, 1, 3))  # zamiana osi (żeby było H x W)
    return reshape(images, rows, cols, 1, n)  # dodanie kanału (C=1)
end

# wczytuje etykiety (klasy)
function read_labels(path)
    io = GzipDecompressorStream(open(path, "r")) 
    magic = read_uint32_be(io)
    @assert magic == 2049 "Invalid label file"  
    n = Int(read_uint32_be(io)) 
    raw = read(io, n) 
    close(io)
    return Int.(raw) .+ 1  # konwersja do Int + przesunięcie (Julia indeksuje od 1)
end

# zamienia etykiety na one-hot encoding
function one_hot(labels, classes)
    N = length(labels)  # liczba próbek
    y = zeros(Float32, classes, N)  # macierz wynikowa
    for i in 1:N
        y[labels[i], i] = 1f0  # ustawienie 1 dla poprawnej klasy
    end
    return y
end

# pobiera plik z internetu jeśli nie ma go lokalnie
function download_file(url, path)
    if !isfile(path)
        Downloads.download(url, path)  # pobranie z internetu
    end
    return path
end

# główna funkcja ładująca FashionMNIST
function load_fashionmnist(data_dir::AbstractString = "data")
    dir = ensure_dir(data_dir)
    train_images_path = joinpath(dir, "train-images-idx3-ubyte.gz")
    train_labels_path = joinpath(dir, "train-labels-idx1-ubyte.gz")
    test_images_path = joinpath(dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = joinpath(dir, "t10k-labels-idx1-ubyte.gz")

    # pobranie plików jeśli ich nie ma
    download_file("$(FASHION_MNIST_BASE)/train-images-idx3-ubyte.gz", train_images_path)
    download_file("$(FASHION_MNIST_BASE)/train-labels-idx1-ubyte.gz", train_labels_path)
    download_file("$(FASHION_MNIST_BASE)/t10k-images-idx3-ubyte.gz", test_images_path)
    download_file("$(FASHION_MNIST_BASE)/t10k-labels-idx1-ubyte.gz", test_labels_path)

    # wczytanie danych i konwersja etykiet na one-hot
    train_X = read_images(train_images_path)
    train_y = one_hot(read_labels(train_labels_path), 10)
    test_X = read_images(test_images_path)
    test_y = one_hot(read_labels(test_labels_path), 10)
    return train_X, train_y, test_X, test_y
end


# iterator batchy (mini-batchy do treningu)
struct BatchIterator
    X
    y
    batch_size::Int
    indices::Vector{Int}
end

# tworzy iterator batchy z opcjonalnym shuffle
function eachbatch(X, y, batch_size::Int; shuffle::Bool = true)
    N = isa(y, Variable) ? size(y.output, 2) : size(X, 4)  # liczba próbek
    indices = collect(1:N)  # indeksy danych
    if shuffle
        Random.shuffle!(indices)  # losowe przemieszanie danych
    end
    return BatchIterator(X, y, batch_size, indices)
end

# implementacja iteracji po batchach
Base.iterate(it::BatchIterator, state=1) = state > length(it.indices) ? nothing : begin
    last = min(state + it.batch_size - 1, length(it.indices))  # koniec batcha
    inds = it.indices[state:last]  # indeksy batcha
    Xdata = isa(it.X, Variable) ? it.X.output : it.X  # obsługa Variable i zwykłych danych
    Ydata = isa(it.y, Variable) ? it.y.output : it.y
    xb = @view Xdata[:, :, :, inds]  # batch wejścia, @view - „nie kopiuj danych, tylko stwórz widok”
    yb = @view Ydata[:, inds]  # batch etykiet
    ((xb, yb), last + 1)  # zwrot batcha i następnego stanu
end

# iterator ma znaną długość
Base.IteratorSize(::Type{BatchIterator}) = Base.HasLength()

# liczba batchy w iteratorze
Base.length(it::BatchIterator) = ceil(Int, length(it.indices) / it.batch_size)