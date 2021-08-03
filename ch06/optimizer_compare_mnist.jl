import Random: shuffle

using Plots
include("../dataset/mnist.jl") # load_mnist
include("../common/util.jl") # smooth_curve
include("../common/multi_layer_net.jl") # MultiLayerNet
include("../common/optimizer.jl")


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

train_size = size(x_train, 1)
batch_size = 128
max_iterations = 2000


# 1:実験の設定==========
optimizers = Dict(
    "SGD"      => SGD(),
    "Momentum" => Momentum(),
    "AdaGrad"  => AdaGrad(),
    "Adam"     => Adam(),
#    "RMSprop" => RMSprop(),
)

networks = Dict()
train_loss = Dict()
for (key,_) = optimizers
    networks[key] = MultiLayerNet(784, [100, 100, 100, 100], 10)
    train_loss[key] = []
end


# 2:訓練の開始==========
for i = 0:max_iterations
    batch_mask = shuffle(1:train_size)[1:batch_size]
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask]
    
    for (key,_) in optimizers
        grads = gradient(networks[key], x_batch, t_batch)
        update(optimizers[key], networks[key].params, grads)
    
        loss_val = loss(networks[key], x_batch, t_batch)
        append!(train_loss[key], loss_val)
    end

    if i % 100 == 0
        println( "=========== iteration: $i ===========")
        for (key,_) = optimizers
            loss_val = loss(networks[key], x_batch, t_batch)
            println("$key : $loss_val")
        end
    end
end


# 3.グラフの描画==========
markers = Dict("SGD"=> :circle, "Momentum"=> :x,      "AdaGrad"=> :rect,  "Adam"=> :diamond)
# colors  = Dict("SGD"=> :blue,   "Momentum"=> :orange, "AdaGrad"=> :green, "Adam"=> :red)

plot(xlabel="iterations", ylabel="loss", ylim=(0, 1))
for (key,_) in optimizers
    y = smooth_curve(train_loss[key])
    x = 0:length(y)-1
    mx = 0:100:length(y)-1
    plot!(x, y, label=key)
    plot!(mx, y[mx.+1], line=false, markershape=markers[key], primary=false)
end
savefig("../image/ch06/fig06-09.png")
