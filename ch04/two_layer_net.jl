include("../common/functions.jl")
include("../common/gradient.jl") # numerical_gradient

mutable struct TwoLayerNet
    params::Dict{String, Matrix{T}} where T <: AbstractFloat
end 

function TwoLayerNet(input_size, hidden_size, output_size, weight_init_std=0.01)
    params = Dict(
        "W1" => weight_init_std * randn(input_size, hidden_size),
        "b1" => zeros(1, hidden_size),
        "W2" => weight_init_std * randn(hidden_size, output_size),
        "b2" => zeros(1, output_size)
    )

    return TwoLayerNet(params)
end

function predict(self::TwoLayerNet, x)
    W1, W2 = self.params["W1"], self.params["W2"]
    b1, b2 = self.params["b1"], self.params["b2"]
    
    a1 = x * W1 .+ b1
    z1 = sigmoid.(a1)
    a2 = z1 * W2 .+ b2
    y = softmax(a2)

    return y
end

# x:入力データ, t:教師データ
function loss(self::TwoLayerNet, x, t)
    y = predict(self, x)    
    return cross_entropy_error(y, t)
end
    
function accuracy(self::TwoLayerNet, x, t)
    y = predict(self, x)
    y = argmax(y, dims=2)
    t = argmax(t, dims=2)
    
    accuracy = sum(y .== t) / size(x, 1)
    return accuracy
end

# x:入力データ, t:教師データ
function numerical_gradient(self::TwoLayerNet, x, t)
    loss_W = (W)->loss(self, x, t)
    
    grads = Dict()
    grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
    
    return grads
end

function gradient(self::TwoLayerNet, x, t)
    W1, W2 = self.params["W1"], self.params["W2"]
    b1, b2 = self.params["b1"], self.params["b2"]
    grads = Dict()
    
    batch_num = size(x, 1)
    
    # forward
    a1 = x * W1 .+ b1
    z1 = sigmoid.(a1)
    a2 = z1 * W2 .+ b2
    y = softmax(a2)
    
    # backward
    dy = (y - t) ./ batch_num
    grads["W2"] = z1' * dy
    grads["b2"] = sum(dy, dims=1)
    
    dz1 = dy * W2'
    da1 = sigmoid_grad(a1) .* dz1
    grads["W1"] = x' * da1
    grads["b1"] = sum(da1, dims=1)

    return grads
end
