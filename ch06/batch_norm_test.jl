include("../dataset/mnist.jl")

import Random: shuffle

import OrderedCollections: OrderedDict
using Plots

import .MNIST: load_mnist
include("../common/multi_layer_net_extend.jl") # MultiLayerNetExtend
include("../common/optimizer.jl") # SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

# 学習データを削減
x_train = x_train[1:1000, :]
t_train = t_train[1:1000]

max_epochs = 20
train_size = size(x_train, 1)
batch_size = 100
learning_rate = 0.01


function __train(weight_init_std)
    bn_network = MultiLayerNetExtend(784, [100, 100, 100, 100, 100], 10, 
                                    weight_init_std=weight_init_std, use_batchnorm=true)
    network = MultiLayerNetExtend(784, [100, 100, 100, 100, 100], 10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(learning_rate)
    
    train_acc_list = zeros(0)
    bn_train_acc_list = zeros(0)
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i = 0:1000000000
        batch_mask = shuffle(1:train_size)[1:batch_size]
        x_batch = x_train[batch_mask, :]
        t_batch = t_train[batch_mask]
    
        for _network = (bn_network, network)
            grads = gradient(_network, x_batch, t_batch)
            update(optimizer, _network.params, grads)
        end
        if i % iter_per_epoch == 0
            train_acc = accuracy(network, x_train, t_train)
            bn_train_acc = accuracy(bn_network, x_train, t_train)
            append!(train_acc_list, train_acc)
            append!(bn_train_acc_list, bn_train_acc)
    
            println("epoch : $epoch_cnt | $train_acc - $bn_train_acc")
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs
                break
            end
        end
    end
    return train_acc_list, bn_train_acc_list
end

# 3.グラフの描画==========
weight_scale_list =  exp10.(range(0, -4, length=16))

p=[]
for (i, w) = enumerate(weight_scale_list)
    x = 1:max_epochs
    xm = 1:2:max_epochs
    println( "============== $(i)/16 ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plot(title="W:$w", ylim=(0.0, 1.0), legend=:bottomright)
    if i == 16
        plot!(x.-1, bn_train_acc_list, label="Batch Normalization")
        plot!(xm.-1, bn_train_acc_list[xm], marker=:circle, line=false, primary=false)
        plot!(x.-1, train_acc_list, line=:dash, label="Normal(without BatchNorm)")
        plot!(xm.-1, train_acc_list[xm], marker=:rect, line=false, primary=false)
    else
        plot!(x.-1, bn_train_acc_list, leg=nothing)
        plot!(xm.-1, bn_train_acc_list[xm], marker=:circle, line=false, primary=false)
        plot!(x.-1, train_acc_list, line=:dash, leg=nothing)
        plot!(xm.-1, train_acc_list[xm], marker=:rect, line=false, primary=false)
    end

    if i % 4 != 1
        plot!(yticks=nothing)
    else
        plot!(ylabel="accuracy")
    end
    if i < 12
        push!(p, plot!(xticks=nothing))
    else
        push!(p, plot!(xlabel="epochs"))
    end
end
plot(p..., layout=(4, 4), size=(1000,1000))
savefig("../image/ch06/fig06-19.png")
