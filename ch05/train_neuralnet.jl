import Random: shuffle
import Printf: @sprintf

include("../dataset/mnist.jl") # load_mnist
include("two_layer_net.jl") # TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=true, one_hot_label=true)

network = TwoLayerNet(784, 50, 10)

iters_num = 10000
train_size = size(x_train, 1)
batch_size = 100
learning_rate = 0.1

train_loss_list = zeros(0)
train_acc_list = zeros(0)
test_acc_list = zeros(0)

iter_per_epoch = max(train_size / batch_size, 1)

for i = 0:iters_num
    batch_mask = shuffle(1:train_size)[1:batch_size]
    x_batch = x_train[batch_mask, :]
    t_batch = t_train[batch_mask, :]
    
    # 勾配
    #grad = numerical_gradient(network, x_batch, t_batch)
    grad = gradient(network, x_batch, t_batch)
    
    # 更新
    for (key, _)=network.params # key=("W1", "b1", "W2", "b2")
        network.params[key] .-= learning_rate * grad[key]
    end
    
    loss_val = loss(network, x_batch, t_batch)
    append!(train_loss_list, loss_val)
    
    if i % iter_per_epoch == 0
        train_acc = accuracy(network, x_train, t_train)
        test_acc = accuracy(network, x_test, t_test)
        append!(train_acc_list, train_acc)
        append!(test_acc_list, test_acc)
        println(@sprintf("%8.4f, %8.4f", train_acc, test_acc))
    end
end
