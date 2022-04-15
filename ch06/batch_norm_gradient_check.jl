include("../dataset/mnist.jl")

import Statistics: mean

import .MNIST: load_mnist
include("../common/multi_layer_net_extend.jl") # MultiLayerNetExtend

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true, one_hot_label=true)

network = MultiLayerNetExtend(784, [100, 100], 10,
                              use_batchnorm=true)

x_batch = x_train[1:1, :]
t_batch = t_train[1:1, :]

grad_backprop = gradient(network, x_batch, t_batch)
grad_numerical = numerical_gradient(network, x_batch, t_batch)


for (key,_) = grad_numerical
    diff = mean( abs.(grad_backprop[key] - grad_numerical[key]) )
    println("$key : $diff")
end