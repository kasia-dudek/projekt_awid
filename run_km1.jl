if !isdefined(Main, :Chain)
    include("src/MiniAD.jl")
    using .MiniAD
    using .MiniAD: DropoutOp
end
using Random

# dla powtarzalnych wyników
Random.seed!(0)

# buduje graf obliczeniowy dla zadanego batcha
function build_graph(model, batch_size)
    x = Variable(zeros(Float32, 28, 28, 1, batch_size), name=:x)
    y = Variable(zeros(Float32, 10, batch_size), name=:y)
    logits = model(x)
    loss = logitcrossentropy(logits, y)
    order = topo_sort(loss)
    dropout_nodes = [node for node in order if node isa OperatorNode && node.f isa DropoutOp]
    return x, y, logits, loss, order, dropout_nodes
end

# resetuje maski dropout przed kolejnym forwardem
function reset_dropout_masks!(dropout_nodes)
    for node in dropout_nodes
        refresh_dropout_mask!(node.f)
    end
end

# liczy stratę i accuracy dla całego zbioru
function loss_and_accuracy(model, X, Y, batch_size, x_var, y_var, logits, loss, order)
    total_loss = 0f0
    total_correct = 0
    total_samples = 0

    eval!(model)  # wyłączenie dropout na czas ewaluacji

    for (xb, yb) in eachbatch(X, Y, batch_size, shuffle=false)
        x_var.output = xb
        y_var.output = yb

        forward!(order)

        batch_logits = value(logits)
        batch_size_actual = size(xb, 4)
        total_loss += value(loss) * batch_size_actual
        total_correct += sum(argmax_classes(batch_logits) .== argmax_classes(yb))
        total_samples += batch_size_actual
    end

    return total_loss / total_samples, total_correct / total_samples
end

# wczytanie FashionMNIST
# układ danych: [wysokość, szerokość, kanały, liczba_próbek]
# etykiety są zapisane jako one-hot
train_X, train_y, test_X, test_y = load_fashionmnist("data")
println("Loaded FashionMNIST: train=$(size(train_X)), test=$(size(test_X))")

# definicja modelu CNN
# architektura:
# 28x28x1 -> Conv -> MaxPool -> Conv -> MaxPool -> Flatten -> Dense -> ReLU -> Dropout -> Dense
model = Chain(
    Conv2D(1, 6, (3, 3), padding=1, bias=false),
    MaxPool2D((2, 2)),
    Conv2D(6, 16, (3, 3), padding=1, bias=false),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(16 * 7 * 7, 84),
    relu,
    Dropout(0.4),
    Dense(84, 10)
)

# hiperparametry
ps = params(model)
batch_size = 10
eval_batch_size = size(test_X,4) 
learning_rate = 0.01f0
num_epochs = 3

# start mierzenia czasu
start_time = time()

println("\n" * "="^80)
println("KM1 FashionMNIST Training Benchmark")
println("="^80)
println("Model: 2-layer CNN with Dropout")
println("Hyperparameters: batch_size=$batch_size, lr=$learning_rate, epochs=$num_epochs")
println("="^80 * "\n")

# budowa grafów do treningu i ewaluacji
train!(model)
train_x_var, train_y_var, train_logits, train_loss, train_order, train_dropout_nodes = build_graph(model, batch_size)

eval!(model)
eval_x_var, eval_y_var, eval_logits, eval_loss, eval_order, _ = build_graph(model, eval_batch_size)

# sprawdzenie accuracy przed treningiem
initial_test_loss, initial_test_acc = loss_and_accuracy(model, test_X, test_y, eval_batch_size, eval_x_var, eval_y_var, eval_logits, eval_loss, eval_order)
println("Pre-training Test Accuracy (random init): $(round(initial_test_acc * 100, digits=2))%\n")

# pętla treningowa
for epoch in 1:num_epochs
    train!(model)  # włączenie trybu treningowego

    # jedna epoka = jedno pełne przejście po danych treningowych
    # dane dzielimy na batche i dla każdego robimy: forward -> backward -> update
    epoch_start = time()
    for (xb, yb) in eachbatch(train_X, train_y, batch_size, shuffle=true)
        train_x_var.output = xb
        train_y_var.output = yb
        reset_dropout_masks!(train_dropout_nodes)

        forward!(train_order)
        backward!(train_order)

        sgd_step!(ps, learning_rate)
    end

    epoch_time = time() - epoch_start
    test_loss, test_acc = loss_and_accuracy(model, test_X, test_y, eval_batch_size, eval_x_var, eval_y_var, eval_logits, eval_loss, eval_order)

    println("Epoch $epoch/$num_epochs | epoch_time=$(round(epoch_time, digits=2))s | test_loss=$(round(test_loss, digits=4)) | test_acc=$(round(test_acc * 100, digits=2))%")
end

# końcowa ewaluacja na train i test
final_train_loss, final_train_acc = loss_and_accuracy(model, train_X, train_y, eval_batch_size, eval_x_var, eval_y_var, eval_logits, eval_loss, eval_order)
final_test_loss, final_test_acc = loss_and_accuracy(model, test_X, test_y, eval_batch_size, eval_x_var, eval_y_var, eval_logits, eval_loss, eval_order)

# koniec mierzenia czasu
end_time = time()
total_time = end_time - start_time

println("\n" * "="^80)
println("KM1 FashionMNIST Training Results")
println("="^80)
println("Pre-training Test Accuracy: $(round(initial_test_acc * 100, digits=2))%")
println("Final Train Loss: $(round(final_train_loss, digits=4))")
println("Final Train Accuracy (full train set): $(round(final_train_acc * 100, digits=2))%")
println("Final Test Loss: $(round(final_test_loss, digits=4))")
println("Final Test Accuracy (full test set): $(round(final_test_acc * 100, digits=2))%")
println("\nTotal End-to-End Benchmark Time: $(round(total_time, digits=2))s")
println("="^80)