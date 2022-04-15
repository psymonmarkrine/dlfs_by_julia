include("../dataset/mnist.jl")

import Random: shuffle
import Printf: @sprintf
using Plots

import .MNIST: load_mnist
include("two_layer_net.jl") # TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true, one_hot_label=true)

network = TwoLayerNet(784, 50, 10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = size(x_train, 1)
batch_size = 100
learning_rate = 0.1

train_loss_list = zeros(0)
train_acc_list = zeros(0)
test_acc_list = zeros(0)

iter_per_epoch = max(train_size / batch_size, 1)

for i in 0:iters_num
    batch_mask = shuffle(1:train_size)[1:batch_size]
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask, :]
    
    # 勾配の計算
    #grad = numerical_gradient(network, x_batch, t_batch)
    grad = gradient(network, x_batch, t_batch)
    
    # パラメータの更新
    for (key, _)=network.params # key=("W1", "b1", "W2", "b2")
        network.params[key] -= learning_rate * grad[key]
    end
    loss_val = loss(network, x_batch, t_batch)
    append!(train_loss_list, loss_val)
    
    if i % iter_per_epoch == 0
        train_acc = accuracy(network, x_train, t_train)
        test_acc = accuracy(network, x_test, t_test)
        append!(train_acc_list, train_acc)
        append!(test_acc_list, test_acc)
        println("train acc, test acc | " * @sprintf("%8.4f, %8.4f", train_acc, test_acc))
    end
end

# グラフの描画
markers = Dict("train"=>:circle, "test"=>:rect)
x = 0:length(train_acc_list)-1
plot(xlabel="epochs", ylabel="accuracy", ylim=(0, 1.0), legend=:bottomright)
plot!(x, train_acc_list, label="train acc")
plot!(x, test_acc_list, label="test acc", linestyle=:dash)
savefig("../image/ch04/fig04-12.png")
