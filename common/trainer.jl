import Random: shuffle
import Printf: @sprintf

include("optimizer.jl")

function getpart(x::Array{T,N}, indexes) where {T,N}
    """Pythonのインデックスっぽく取り出すための関数
    ```
    julia> x = rand(10, 3, 28, 28);

    julia> size(x)
    (10, 3, 28, 28)

    julia> size(getpart(x, (2:5, :, 3)))
    (4, 3, 28)

    julia> size(getpart(x, (2:5,)))
    (4, 3, 28, 28)
    ```
    """
    l = length(indexes)
    d = ndims(x)
    return x[[i>l ? (:) : indexes[i] for i=1:d]...]
end

mutable struct Trainer
    network
    verbose
    x_train
    t_train
    x_test
    t_test
    epochs
    batch_size
    evaluate_sample_num_per_epoch
    optimizer
    train_size
    iter_per_epoch
    max_iter
    current_iter
    current_epoch
    train_loss_list
    train_acc_list
    test_acc_list
end

"""ニューラルネットの訓練を行うクラス
"""
function Trainer(network, x_train, t_train, x_test, t_test;
                 epochs=20, mini_batch_size=100,
                 optimizer="SGD", optimizer_param=(lr=0.01,), 
                 evaluate_sample_num_per_epoch=nothing, verbose=true)
    # optimizer
    optimizer_class_dict = Dict("sgd"=>SGD, "momentum"=>Momentum, "nesterov"=>Nesterov,
    "adagrad"=>AdaGrad, "rmsprop"=>RMSprop, "adam"=>Adam)
    optimizer = optimizer_class_dict[lowercase(optimizer)](optimizer_param...)

    train_size = size(x_train, 1)
    iter_per_epoch = max(Integer(floor(train_size / mini_batch_size)), 1)
    max_iter = Integer(epochs * iter_per_epoch)

    return Trainer(network, verbose, x_train, t_train, x_test, t_test, 
                   epochs, mini_batch_size, evaluate_sample_num_per_epoch, optimizer,
                   train_size, iter_per_epoch, max_iter, 0, 0,
                   zeros(0), zeros(0),  zeros(0)
                  )
end

function train_step(self::Trainer)
    batch_mask = shuffle(1:self.train_size)[1:self.batch_size]
    x_batch = getpart(self.x_train, [batch_mask])
    t_batch = getpart(self.t_train, [batch_mask])
    
    grads = gradient(self.network, x_batch, t_batch)
    update(self.optimizer, self.network.params, grads)
    
    loss_val = loss(self.network, x_batch, t_batch)
    append!(self.train_loss_list, loss_val)
    if self.verbose
        println("train loss : $loss_val")
    end
    
    if (self.current_iter % self.iter_per_epoch) == 0
        self.current_epoch += 1
        
        x_train_sample, t_train_sample = self.x_train, self.t_train
        x_test_sample, t_test_sample = self.x_test, self.t_test
        if !isnothing(self.evaluate_sample_num_per_epoch)
            t = self.evaluate_sample_num_per_epoch
            x_train_sample = getpart(self.x_train, [1:t])
            t_train_sample = getpart(self.t_train, [1:t])
            x_test_sample = getpart(self.x_test, [1:t])
            t_test_sample = getpart(self.t_test, [1:t])
        end
        train_acc = accuracy(self.network, x_train_sample, t_train_sample)
        test_acc = accuracy(self.network, x_test_sample, t_test_sample)
        append!(self.train_acc_list, train_acc)
        append!(self.test_acc_list, test_acc)

        if self.verbose
            println("=== epoch : $(self.current_epoch), train acc : $(@sprintf("%8.04f", train_acc)), test acc : $(@sprintf("%8.04f", test_acc)) ===")
        end
    end
    self.current_iter += 1
end

function train(self::Trainer)
    for i = 1:self.max_iter
        train_step(self)
    end
    test_acc = accuracy(self.network, self.x_test, self.t_test)

    if self.verbose
        println("=============== Final Test Accuracy ===============")
        println("test acc : $test_acc")
    end
end