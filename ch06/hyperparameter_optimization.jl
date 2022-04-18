include("../dataset/mnist.jl")
include("../common/commons.jl")

using Plots

import .MNIST: load_mnist
import .Util: shuffle_dataset
using  .MultiLayerNets
using  .Trainers


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

# 高速化のため訓練データの削減
x_train = x_train[1:500, :]
t_train = t_train[1:500]

# 検証データの分離
validation_rate = 0.20
validation_num = Integer(round(size(x_train, 1) * validation_rate))
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[1:validation_num, :]
t_val = t_train[1:validation_num]
x_train = x_train[validation_num:end, :]
t_train = t_train[validation_num:end]


function __train(lr, weight_decay, epocs=50)
    network = MultiLayerNet(784, [100, 100, 100, 100, 100, 100], 10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer="sgd", optimizer_param=(lr=lr,), verbose=false)
    train(trainer)

    return trainer.test_acc_list, trainer.train_acc_list
end

# ハイパーパラメータのランダム探索======================================
optimization_trial = 100
results_val = Dict()
results_train = Dict()

urand = (l=0,u=1)->rand()*(u-l)+l
for _ in 1:optimization_trial
    # 探索したハイパーパラメータの範囲を指定===============
    weight_decay = exp10(urand(-8, -4))
    lr = exp10(urand(-6, -2))
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    println("val acc : $(val_acc_list[end]) | lr : $lr, weight decay : $weight_decay")
    key = "lr:$lr, weight decay:$weight_decay"
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list
end

# グラフの描画========================================================
println("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = Integer(ceil(graph_draw_num / col_num))
i = 0

p = []
for (key, val_acc_list) = sort(collect(results_val), by=(x)->(x[2][end]), rev=true)
    println("Best-$(i+1) (val acc : $(val_acc_list[end])) | $key")

    plot(title="Best-$i", xticks=nothing, ylim=(0.0, 1.0), leg=false)
    if i % 5 != 0
        plot!(yticks=nothing)
    end
    x = 1:length(val_acc_list)
    plot!(x, val_acc_list)
    push!(p, plot!(x, results_train[key], line=:dash))
    global i += 1

    if i >= graph_draw_num
        break
    end
end
plot(p..., layout=(row_num, col_num), size=(1000,1200))
savefig("../image/ch06/fig06-24.png")
