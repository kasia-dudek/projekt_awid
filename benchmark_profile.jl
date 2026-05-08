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
    return x, y, logits, loss, order, dropout_nodes
end

function reset_dropout_masks!(dropout_nodes)
    for node in dropout_nodes
        refresh_dropout_mask!(node.f)
    end
end

function main()
    train_X, train_y, test_X, test_y = load_fashionmnist("data")
    model = Chain(
        Conv2D(1, 6, (3, 3), padding=1, bias=false),
        MaxPool2D((2, 2)),
        Conv2D(6, 16, (3, 3), padding=1, bias=false),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(16 * 7 * 7, 84),
        relu,
        Dropout(0.2),
        Dense(84, 10)
    )

    ps = params(model)
    batch_size = parse(Int, get(ENV, "PROFILE_BATCH_SIZE", "10"))
    train_batches = parse(Int, get(ENV, "PROFILE_TRAIN_BATCHES", "200"))
    eval_batches = parse(Int, get(ENV, "PROFILE_EVAL_BATCHES", "200"))
    lr = 0.01f0

    train!(model)
    x_var, y_var, _, _, order, dropout_nodes = build_graph(model, batch_size)

    forward_t = 0.0
    backward_t = 0.0
    update_t = 0.0
    total_train_t = @elapsed begin
        cnt = 0
        for (xb, yb) in eachbatch(train_X, train_y, batch_size, shuffle=false)
            x_var.output = xb
            y_var.output = yb
            reset_dropout_masks!(dropout_nodes)

            forward_t += @elapsed forward!(order)
            backward_t += @elapsed backward!(order)
            update_t += @elapsed sgd_step!(ps, lr)

            cnt += 1
            cnt >= train_batches && break
        end
    end

    eval!(model)
    eval_x, eval_y, _, _, eval_order, _ = build_graph(model, batch_size)
    eval_t = @elapsed begin
        cnt = 0
        for (xb, yb) in eachbatch(test_X, test_y, batch_size, shuffle=false)
            eval_x.output = xb
            eval_y.output = yb
            forward!(eval_order)
            cnt += 1
            cnt >= eval_batches && break
        end
    end

    println("PROFILE train_batches=$(train_batches) eval_batches=$(eval_batches) batch_size=$(batch_size)")
    println("forward_s=$(round(forward_t, digits=3))")
    println("backward_s=$(round(backward_t, digits=3))")
    println("update_s=$(round(update_t, digits=3))")
    println("train_total_s=$(round(total_train_t, digits=3))")
    println("eval_s=$(round(eval_t, digits=3))")
end

main()
