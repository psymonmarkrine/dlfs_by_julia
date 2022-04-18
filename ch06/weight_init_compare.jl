include("../dataset/mnist.jl")
include("../common/commons.jl")

import Random: shuffle

import OrderedCollections: OrderedDict
using Plots

import .MNIST: load_mnist
import .Util: smooth_curve
import .Optimizer: SGD, update
using  .MultiLayerNets


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

train_size = size(x_train, 1)
batch_size = 128
max_iterations = 2000


# 1:実験の設定==========
weight_init_types = OrderedDict("std=0.01"=>0.01, "Xavier"=>"sigmoid", "He"=>"relu")
optimizer = SGD(0.01)

networks = Dict()
train_loss = Dict()
for (key, weight_type) = weight_init_types
    networks[key] = MultiLayerNet(784, [100, 100, 100, 100], 10, weight_init_std=weight_type)
    train_loss[key] = zeros(0)
end

# 2:訓練の開始==========
for i = 0:max_iterations
    batch_mask = shuffle(1:train_size)[1:batch_size]
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask]
    
    for (key,_) = weight_init_types
        grads = gradient(networks[key], x_batch, t_batch)
        update(optimizer, networks[key].params, grads)
    
        loss_val = loss(networks[key], x_batch, t_batch)
        append!(train_loss[key], loss_val)
    end

    if i % 100 == 0
        println("=========== iteration : $i ===========")
        for (key,_) = weight_init_types
            loss_val = loss(networks[key], x_batch, t_batch)
            println("$key : $loss_val")
        end
    end
end


# 3.グラフの描画==========
markers = Dict("std=0.01"=> :circle, "Xavier"=> :rect, "He"=> :diamond)
plot(xlabel="iterations", ylabel="loss", ylim=(0, 2.5))
for (key,_) = weight_init_types
    y = smooth_curve(train_loss[key])
    x = 0:length(y)-1
    mx = 0:100:length(y)-1
    plot!(x, y, label=key)
    plot!(mx, y[mx.+1], line=false, markershape=markers[key], primary=false)
end
savefig("../image/ch06/fig06-15.png")
