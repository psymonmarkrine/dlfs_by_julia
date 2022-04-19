include("../dataset/mnist.jl")
include("simple_convnet.jl")

using Plots

import .MNIST: load_mnist
using  .Trainers


# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=false)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 10

network = SimpleConvNet((1,28,28), 
                        Dict("filter_num"=> 30, "filter_size"=> 5, "pad"=> 0, "stride"=> 1),
                        100, 10, 0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  gradient=gradient, loss=loss, accuracy=accuracy,
                  epochs=max_epochs, mini_batch_size=2000,
                  optimizer="Adam", optimizer_param=(lr=0.02),
                  evaluate_sample_num_per_epoch=1000)
train(trainer)

# パラメータの保存
save_params(network, "params.h5")
println("Saved Network Parameters!")

# グラフの描画
markers = Dict("train"=> :circle, "test"=> :rect)
x = 1:max_epochs
xm = 1:2:max_epochs
plot(xlabel="epochs", ylabel="accuracy", ylim=(0, 1.0), legend=:bottomright)
plot!(x.-1, trainer.train_acc_list, label="train")
plot!(xm.-1, trainer.train_acc_list[xm], marker=:circle, line=false, primary=false)
plot!(x.-1, trainer.test_acc_list, label="test")
plot!(xm.-1, trainer.test_acc_list[xm], marker=:rect, line=false, primary=false)
savefig("../image/ch07/fig07.png")
