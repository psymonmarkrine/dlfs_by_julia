include("../dataset/mnist.jl")
include("../common/commons.jl")

import Random: shuffle

import OrderedCollections: OrderedDict
using Plots

import .MNIST: load_mnist
import .Optimizer: SGD, update
using  .MultiLayerNets


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

# 過学習を再現するために、学習データを削減
x_train = x_train[1:300, :]
t_train = t_train[1:300]

# weight decay（荷重減衰）の設定 =======================
# weight_decay_lambda = 0.0 # weight decayを使用しない場合
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(784, [100, 100, 100, 100, 100, 100], 10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(0.01)

max_epochs = 201
train_size = size(x_train, 1)
batch_size = 100

train_loss_list = zeros(0)
train_acc_list = zeros(0)
test_acc_list = zeros(0)

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i = 0:1000000000
    batch_mask = shuffle(1:train_size)[1:batch_size]
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask]

    grads = gradient(network, x_batch, t_batch)
    update(optimizer, network.params, grads)

    if i % iter_per_epoch == 0
        train_acc = accuracy(network, x_train, t_train)
        test_acc = accuracy(network, x_test, t_test)
        append!(train_acc_list, train_acc)
        append!(test_acc_list, test_acc)

        println("epoch : $epoch_cnt, train acc : $train_acc, test acc : $test_acc")

        global epoch_cnt += 1
        if epoch_cnt >= max_epochs
            break
        end
    end
end


# 3.グラフの描画==========
markers = Dict("train"=> :circle, "test"=> :rect)
x = 1:max_epochs
xm = 1:10:max_epochs
plot(xlabel="epochs", ylabel="accuracy", ylim=(0, 1.0), legend=:bottomright)
plot!(x.-1, train_acc_list, label="train")
plot!(xm.-1, train_acc_list[xm], marker=:circle, line=false, primary=false)
plot!(x.-1, test_acc_list, label="test")
plot!(xm.-1, test_acc_list[xm], marker=:rect, line=false, primary=false)
savefig("../image/ch06/fig06-21.png")
