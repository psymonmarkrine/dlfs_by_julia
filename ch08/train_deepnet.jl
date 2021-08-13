using Plots
include("../dataset/mnist.jl") # load_mnist
include("deep_convnet.jl") # DeepConvNet
include("../common/trainer.jl") # Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=false)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=10, mini_batch_size=1000,
                  optimizer="Adam", optimizer_param=(lr=0.001,),
                  evaluate_sample_num_per_epoch=1000)
train(trainer)

# パラメータの保存
save_params(network, "deep_convnet_params.h5")
println("Saved Network Parameters!")
