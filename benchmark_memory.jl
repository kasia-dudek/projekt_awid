if !isdefined(Main, :Chain)
    include("src/MiniAD.jl")
    using .MiniAD
    using .MiniAD: DropoutOp
end
using Random

Random.seed!(0)

function build_graph(model, batch_size)
    x = Variable(zeros(Float32, 28, 28, 1, batch_size), name=:x)
    y = Variable(zeros(Float32, 10, batch_size), name=:y)
    logits = model(x)
    loss = logitcrossentropy(logits, y)
    order = topo_sort(loss)
    dropout_nodes = [node for node in order if node isa OperatorNode && node.f isa DropoutOp]
    return x, y, order, dropout_nodes
end

function reset_dropout_masks!(dropout_nodes)
    for node in dropout_nodes
        MiniAD.refresh_dropout_mask!(node.f)
    end
end

function build_model(dropout_p::Float32)
    return Chain(
        Conv2D(1, 6, (3, 3), padding=1, bias=false),
        MaxPool2D((2, 2)),
        Conv2D(6, 16, (3, 3), padding=1, bias=false),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(16 * 7 * 7, 84),
        relu,
        Dropout(dropout_p),
        Dense(84, 10)
    )
end

function measure_allocations(; train_batches=100, eval_batches=100, batch_size=10, dropout_p=0.2f0)
    train_X, train_y, test_X, test_y = load_fashionmnist("data")
    model = build_model(dropout_p)
    ps = params(model)

    train!(model)
    x_var, y_var, train_order, dropout_nodes = build_graph(model, batch_size)
    train_alloc = @allocated begin
        c = 0
        for (xb, yb) in eachbatch(train_X, train_y, batch_size, shuffle=false)
            x_var.output = xb
            y_var.output = yb
            reset_dropout_masks!(dropout_nodes)
            forward!(train_order)
            backward!(train_order)
            sgd_step!(ps, 0.01f0)
            c += 1
            c >= train_batches && break
        end
    end

    eval!(model)
    ex_var, ey_var, eval_order, _ = build_graph(model, batch_size)
    eval_alloc = @allocated begin
        c = 0
        for (xb, yb) in eachbatch(test_X, test_y, batch_size, shuffle=false)
            ex_var.output = xb
            ey_var.output = yb
            forward!(eval_order)
            c += 1
            c >= eval_batches && break
        end
    end

    println("MEM_ALLOC train100_bytes=$(train_alloc)")
    println("MEM_ALLOC eval100_bytes=$(eval_alloc)")
end

measure_allocations()
