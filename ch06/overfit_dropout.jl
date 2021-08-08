using Plots
include("../dataset/mnist.jl") # load_mnist
include("../common/multi_layer_net_extend.jl") # MultiLayerNetExtend
include("../common/trainer.jl") # Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true)

# 過学習を再現するために、学習データを削減
x_train = x_train[1:300,:]
t_train = t_train[1:300]

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = true  # Dropoutなしのときの場合はfalseに
dropout_ratio = 0.15
# ====================================================

network = MultiLayerNetExtend(784, [100, 100, 100, 100, 100, 100], 10,
                              use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer="sgd", optimizer_param=(lr=0.01,), verbose=true)
train(trainer)

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# グラフの描画==========
markers = Dict("train"=> :circle, "test"=> :rect)
len = length(train_acc_list)
x = 1:len
xm = 1:10:len
plot(xlabel="epochs", ylabel="accuracy", xlim=(0, len), ylim=(0, 1.0), legend=:bottomright)
plot!(x.-1,  train_acc_list[x], label="train")
plot!(xm.-1, train_acc_list[xm], marker=:circle, line=false, primary=false)
plot!(x.-1,  test_acc_list[x], label="test")
plot!(xm.-1, test_acc_list[xm], marker=:rect, line=false, primary=false)
savefig("../image/ch06/fig06-23b.png")
