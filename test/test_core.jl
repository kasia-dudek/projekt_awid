# małe testy poprawności

include("../src/MiniAD.jl")
using .MiniAD
using Test

@testset "Core AD tests" begin

    @testset "f(x) = x * x" begin
        x = Variable(3.0, name=:x)
        y = x * x

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test value(y) == 9.0
        @test grad(x) == 6.0
    end

    @testset "f(x, y) = x * y + x" begin
        x = Variable(2.0, name=:x)
        y = Variable(4.0, name=:y)
        z = x * y + x

        g = topo_sort(z)
        forward!(g)
        backward!(g)

        @test value(z) == 10.0
        @test grad(x) == 5.0
        @test grad(y) == 2.0
    end

end

@testset "New features tests" begin

    @testset "Dense forward/backward" begin
        layer = Dense(3, 2)
        x = Variable(randn(3), name=:x)
        y = layer(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (2,)
        @test size(grad(layer.W)) == (2, 3)
        @test size(grad(layer.b)) == (2,)
    end

    @testset "Flatten" begin
        f = Flatten()
        x = Variable(randn(2, 3, 1, 1), name=:x)  # HWCN
        y = f(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (6, 1)
        @test size(grad(x)) == (2, 3, 1, 1)
    end

    @testset "Chain" begin
        model = Chain(Dense(3, 4), relu, Dense(4, 2))
        x = Variable(randn(3, 1), name=:x)
        y = model(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (2, 1)
        ps = params(model)
        @test length(ps) == 4  # 2 macierze W i 2 biasy
    end

    @testset "logitcrossentropy" begin
        logits = Variable([1.0, 2.0, 3.0], name=:logits)
        labels = [0.0, 0.0, 1.0]
        loss = logitcrossentropy(logits, labels)

        g = topo_sort(loss)
        forward!(g)
        backward!(g)

        @test value(loss) > 0
        @test size(grad(logits)) == (3,)
    end

    @testset "Full training step" begin
        model = Chain(Dense(3, 2), relu, Dense(2, 3))
        x = Variable(randn(3, 1), name=:x)
        logits = model(x)
        labels = [0.0 1.0 0.0]'
        loss = logitcrossentropy(logits, labels)

        g = topo_sort(loss)
        forward!(g)
        backward!(g)

        ps = params(model)
        loss_before = value(loss)
        sgd_step!(ps, 0.01)

        forward!(g)
        @test value(loss) != loss_before
    end

    @testset "Softmax forward/backward" begin
        # test softmax dla pojedynczego wektora
        x_1d = Variable([1.0, 2.0, 3.0], name=:x)
        y_1d = softmax(x_1d)
        g_1d = topo_sort(y_1d)
        forward!(g_1d)
        @test abs(sum(value(y_1d)) - 1.0) < 1e-6
        @test all(value(y_1d) .> 0)

        # test softmax dla batcha
        x_2d = Variable(randn(3, 2), name=:x)
        y_2d = softmax(x_2d)
        g_2d = topo_sort(y_2d)
        forward!(g_2d)
        @test size(value(y_2d)) == (3, 2)
        @test all(value(y_2d) .> 0)
        for i in 1:2
            @test abs(sum(value(y_2d)[:, i]) - 1.0) < 1e-6
        end

        backward!(g_2d)
        @test size(grad(x_2d)) == (3, 2)
    end

    @testset "Conv2D forward/backward" begin
        x = Variable(randn(5, 5, 1, 1), name=:x)  # HWCN
        conv = Conv2D(1, 2, (3, 3), padding=1)
        y = conv(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (5, 5, 2, 1)
        @test size(grad(conv.filters)) == (3, 3, 1, 2)
        @test size(grad(conv.bias)) == (2,)
        @test size(grad(x)) == (5, 5, 1, 1)
    end

    @testset "Conv2D numeric forward/backward" begin
        x = Variable(reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1, 1), name=:x)
        conv = Conv2D(1, 1, (2, 2), padding=0)
        conv.filters.output .= reshape([1.0, 0.0, 0.0, 1.0], 2, 2, 1, 1)
        conv.bias.output .= [0.0]

        y = conv(x)
        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (1, 1, 1, 1)
        @test value(y)[1] == 5.0
        @test size(grad(x)) == (2, 2, 1, 1)
        @test grad(x) == reshape([1.0, 0.0, 0.0, 1.0], 2, 2, 1, 1)
    end

    @testset "MaxPool2D forward/backward" begin
        x = Variable(randn(4, 4, 1, 1), name=:x)
        pool = MaxPool2D((2, 2))
        y = pool(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (2, 2, 1, 1)
        @test size(grad(x)) == (4, 4, 1, 1)
    end

    @testset "Dropout train/eval" begin
        layer = Dropout(0.5)
        x = Variable(ones(4, 1), name=:x)
        train!(layer)
        y = layer(x)

        g = topo_sort(y)
        forward!(g)
        backward!(g)

        @test size(value(y)) == (4, 1)
        @test size(grad(x)) == (4, 1)
        @test any(value(y) .== 0.0)  # w train część wartości powinna być wyzerowana

        eval!(layer)
        y_eval = layer(x)
        h = topo_sort(y_eval)
        forward!(h)
        @test value(y_eval) == value(x)  # w eval dropout nic nie zmienia
    end

    @testset "Batching" begin
        X = randn(28, 28, 1, 10)
        y = zeros(10, 10)
        for i in 1:10
            y[i, i] = 1.0
        end
        batches_list = collect(eachbatch(X, y, 4, shuffle=false))
        @test length(batches_list) == 3
        @test size(batches_list[1][1]) == (28, 28, 1, 4)
        @test size(batches_list[1][2]) == (10, 4)
    end

    @testset "FashionMNIST loader" begin
        try
            train_X, train_y, test_X, test_y = load_fashionmnist("data")
            @test size(train_X) == (28, 28, 1, 60000)
            @test size(train_y) == (10, 60000)
            @test size(test_X) == (28, 28, 1, 10000)
            @test size(test_y) == (10, 10000)
        catch e
            @info "FashionMNIST loader skipped: $e"
        end
    end

    @testset "Small FashionMNIST model forward" begin
        model = Chain(
            Conv2D(1, 2, (3, 3), padding=1),
            relu,
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(2 * 14 * 14, 3)
        )
        x = Variable(randn(28, 28, 1, 2), name=:x)
        logits = model(x)

        g = topo_sort(logits)
        forward!(g)

        @test size(value(logits)) == (3, 2)
    end

    @testset "Small FashionMNIST training step" begin
        model = Chain(
            Conv2D(1, 2, (3, 3), padding=1),
            relu,
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(2 * 14 * 14, 3)
        )
        x = Variable(randn(28, 28, 1, 2), name=:x)
        labels = zeros(3, 2)
        labels[1, 1] = 1.0
        labels[3, 2] = 1.0
        logits = model(x)
        loss = logitcrossentropy(logits, labels)

        g = topo_sort(loss)
        forward!(g)
        backward!(g)

        ps = params(model)
        before = deepcopy(ps[1].output)
        sgd_step!(ps, 0.01)

        @test any(ps[1].output .!= before)
    end

    @testset "CNN Chain smoke" begin
        model = Chain(
            Conv2D(1, 2, (3, 3), padding=1),
            relu,
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(8, 3)
        )

        x = Variable(randn(4, 4, 1, 1), name=:x)
        logits = model(x)
        labels = [0.0, 1.0, 0.0]
        loss = logitcrossentropy(logits, labels)

        g = topo_sort(loss)
        forward!(g)
        backward!(g)

        ps = params(model)
        loss_before = value(loss)
        sgd_step!(ps, 0.01)

        forward!(g)
        @test value(loss) != loss_before
    end

end