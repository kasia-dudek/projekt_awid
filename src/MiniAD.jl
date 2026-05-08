# główny moduł, który scala resztę

module MiniAD

include("core.jl")
include("ops_scalar.jl")
include("ops_array.jl")
include("utils.jl")
include("dataset.jl")
include("layers.jl")
include("cnn.jl")
include("classifier.jl")
include("training.jl")

# export tego, co będzie dostępne na zewnątrz modułu
export GraphNode, Variable, Constant, OperatorNode
export topo_sort, forward!, backward!, zero_grad!
export value, grad
export Dense, Flatten, Chain, Dropout, Conv2D, MaxPool2D, relu
export logsumexp, logitcrossentropy, softmax
export params, sgd_step!, train!, eval!
export load_fashionmnist, eachbatch, argmax_classes, accuracy
export refresh_dropout_mask!

end