include("../dataset/mnist.jl")

include("deep_convnet.jl") # DeepConvNet

import .MNIST: load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=false)

network = DeepConvNet()
load_params(network, "deep_convnet_params.h5")

sampled = 10000 # 高速化のため
x_test = x_test[1:sampled, :, :, :]
t_test = t_test[1:sampled]

println("caluculate accuracy (float64) ... ")
println(accuracy(network, x_test, t_test, 1000))

# float16に型変換
x_test = Float16.(x_test)
for (_,param) = network.params
    param .= Float16.(param)
end

println("caluculate accuracy (float16) ... ")
println(accuracy(network, x_test, t_test, 1000))
