import Statistics: mean
include("../dataset/mnist.jl") # load_mnist
include("two_layer_net.jl") # TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true, one_hot_label=true)

network = TwoLayerNet(784, 50, 10)

x_batch = x_train[1:3,:]
t_batch = t_train[1:3,:]

grad_numerical = numerical_gradient(network, x_batch, t_batch)
grad_backprop = gradient(network, x_batch, t_batch)

for (key,_) in grad_numerical
    diff = mean( abs.(grad_backprop[key] - grad_numerical[key]) )
    println("$key : $diff")
end
